from datetime import datetime
import torch
import torch.nn as nn
import os
from util.config import *
from architecture.generalizeable_dynamic_field import *
from architecture.encoders import *
from architecture.flow_feature_aggregation import *
from dataloader import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.ray_helpers import *
from util.torch_utils import *
# from util.util import group_indices
from einops import *
from loss import *
import wandb
import copy
import torch
import torch.autograd as autograd
from torchviz import make_dot
from util import positional_encoding, entropy, L2_norm, normalize, L1_norm, mse2psnr
from architecture.mononerf import *
from torch.profiler import profile, record_function, ProfilerActivity
from util.render_utils import raw2outputs,batchify_rays,render_rays

def check_parameters_changed(model, prev_parameters):
    current_parameters = copy.deepcopy(model.state_dict())
    if prev_parameters is None:
        return True
    else:
        for param_name in prev_parameters.keys():
            if not torch.equal(prev_parameters[param_name], current_parameters[param_name]):
                return True
        return False


# PYZSHCOMPLETE_OK
def check_parameters_changed(model, prev_parameters):
    current_parameters = copy.deepcopy(model.state_dict())
    if prev_parameters is None:
        return True
    else:
        for param_name in prev_parameters.keys():
            if not torch.equal(prev_parameters[param_name], current_parameters[param_name]):
                return True
        return False


def calculate_blending_regularization(blending_weights, z_vals, disp, depth_epsilon, temperature=1e-5):
    blending_weights = torch.sigmoid(blending_weights)
    z_vals = 2 / (torch.clamp(z_vals, min=-1., max=1-1e-3) - 1)

    z_vals = normalize(z_vals)

    disp = normalize(disp).unsqueeze(-1)
    lower_bound = disp - depth_epsilon
    upper_bound = disp + depth_epsilon
    bound = torch.logical_or((z_vals < lower_bound),
                             (z_vals > upper_bound)).float()
    return L2_norm(blending_weights*bound)


def compute_depth_loss(dyn_depth, gt_depth):
    dyn_depth_norm = normalize(dyn_depth)
    gt_depth_norm = normalize(gt_depth)

    return L2_norm(dyn_depth_norm - gt_depth_norm)


def compute_mask_flow_loss(blending_pre, blending_cur, blending_post):
    blending_pre, blending_cur, blending_post = normalize(blending_pre), normalize(blending_cur), normalize(blending_post)
    blending_pre, blending_cur, blending_post = torch.sigmoid(blending_pre), torch.sigmoid(blending_cur), torch.sigmoid(blending_post)
    return L2_norm(blending_pre - blending_cur) + L2_norm(blending_cur - blending_post) + L2_norm(blending_pre - blending_post)

def calculate_losses(res, target, fwd_flow, fwd_flow_mask, bwd_flow, bwd_flow_mask, disparity, pose, intrinsics, rays_d, criterion, config):
    fwd_flow_loss = supervise_flows(res["trajectory_2d"][:,1],res["trajectory_2d"][:,2],res["output_full"][...,3],fwd_flow,fwd_flow_mask,pose[:,2],intrinsics[:,2],res["z_vals"],rays_d,criterion)
    bwd_flow_loss = supervise_flows(res["trajectory_2d"][:,1],res["trajectory_2d"][:,0],res["output_full"][...,3],bwd_flow,bwd_flow_mask,pose[:,0],intrinsics[:,0],res["z_vals"],rays_d,criterion)
                            
    l_bw, l_curr, l_fw = criterion(res['rgb_pre'], target), criterion(res['rgb_cur'], target), criterion(res['rgb_post'], target)
    L_corr = (l_curr + l_bw + l_fw)/3
    rgb_loss = criterion(res['rgb_full'], target)
    
    rgb_loss_psnr = mse2psnr(rgb_loss)
    rgb_loss_corr_psnr = mse2psnr(L_corr)
    rgb_pre_psnr, rgb_cur_psnr, rgb_post_psnr = mse2psnr(l_bw), mse2psnr(l_curr), mse2psnr(l_fw)
    
    disparity_loss = compute_depth_loss(
        res['depth_map_full'], -disparity)
    sparse_loss = entropy(res['weights_full']) 
    slow_loss = L1_norm(fwd_flow) + L1_norm(bwd_flow)

    loss = rgb_loss*config.loss.rgb_lambda + L_corr*config.loss.correlation_lambda + \
        disparity_loss*config.loss.disparity_loss_lambda + \
        sparse_loss*config.loss.sparse_loss_lambda + \
        (fwd_flow_loss + bwd_flow_loss) * config.loss.nerf_flow_loss_lambda + \
        slow_loss*config.loss.slow_loss_lambda 

    return {
        "rgb_loss": rgb_loss,
        "L_corr": L_corr,
        "disparity_loss": disparity_loss,
        "sparse_loss": sparse_loss,
        "fwd_flow_loss": fwd_flow_loss,
        "bwd_flow_loss": bwd_flow_loss,
        "rgb_loss_psnr": rgb_loss_psnr,
        "rgb_loss_corr_psnr": rgb_loss_corr_psnr,
        "rgb_cur": l_curr,
        "rgb_pre": l_bw,
        "rgb_post": l_fw,
        "rgb_cur_psnr": rgb_cur_psnr,
        "rgb_pre_psnr": rgb_pre_psnr,
        "rgb_post_psnr": rgb_post_psnr,
        "total_loss": loss
    }

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

def run_net(*args, **kwargs):
    print("hey that's strange")
    pass

def train_dynamic(run,models:dict,
          train_loader,
          optimizer_group:OptimizerGroup,
          criterion,
          config,
          device):
    torch.autograd.set_detect_anomaly(True)

    dir_path = f"{config.checkpoint_dir}/{run.name}"
    os.makedirs(dir_path, exist_ok=True)
    # save config in checkpoint dir
    with open(f"{dir_path}/config.json", "w") as f:
        json.dump(config.to_dict(), f)
    for model in models:
        models[model].train()
    total_batches = len(train_loader)*config.training.epochs
    video_embedding = train_loader.dataset.get_video_embedding().to(device)
    step = 0
    with tqdm(total=total_batches, desc="Training") as pbar:
        for _ in range(config.training.epochs):
            for _, (index, rays_o, rays_d,pose,intrinsics, masks, disparity, fwd_flow, fwd_flow_mask, bwd_flow, bwd_flow_mask, target) in enumerate(train_loader):
                # if step % config.rendering.render_every == 0:
                #     s = np.random.randint(0, train_loader.dataset.n_scenes)
                #     t = np.random.randint(1, train_loader.dataset.n_images-1)
                #     results = batchify_rays(train_loader, s, t, models, video_embedding, config, chunk=config.rendering.chunk, verbose=True, perturb=0, N_samples=64)
                    
                #     rgb = results["rgb_full"].reshape(train_loader.dataset.h, train_loader.dataset.w, 3)
                #     gt_image = train_loader.dataset.images[s,t]*train_loader.dataset.images_masks[s,t]
                #     images = torch.cat([rgb.permute(2,0,1).unsqueeze(0),gt_image.to(device=device).unsqueeze(0)],dim=0)
                #     image = wandb.Image(
                #         images ,
                #         caption=f"Rendered image, scene {s} image {t}"
                #     )
                #     wandb.log({"render": image}, step=step)
                
                loss = 0
                target = target.to(device)
                optimizer_group.zero_grad()
                scene_index, image_indices, _,_ = index
                tensors_to_device = [scene_index,image_indices,rays_o, rays_d,pose,intrinsics, masks, disparity, fwd_flow, fwd_flow_mask, bwd_flow, bwd_flow_mask]
                scene_index,image_indices,rays_o, rays_d,pose,intrinsics, masks, disparity, fwd_flow, fwd_flow_mask, bwd_flow, bwd_flow_mask = map(lambda x: x.to(device), tensors_to_device)
                if (masks==0).all():
                    continue
                masks = masks.unsqueeze(-1)
                target = target*masks  
                
                f_temp = models['video_downsampler'](video_embedding)
                res = render_rays(rays_o, rays_d, models, f_temp, pose, intrinsics, image_indices,scene_index,config, chunk=config.rendering.chunk, verbose=False, perturb=1, N_samples=64,all_parts=True)                
                losses = calculate_losses(res, target, fwd_flow, fwd_flow_mask, bwd_flow, bwd_flow_mask, disparity, pose, intrinsics, rays_d, criterion, config)
                wandb.log(losses)
                wandb.log({
                    "lr_general": optimizer_group.get_lr()[0],
                    "lr_ray_bender": optimizer_group.get_lr()[1],
                    "batch_loss": losses["total_loss"].item()
                })
                loss = losses['total_loss']
                loss.backward()
                optimizer_group.step()
                optimizer_group.scheduler_step()
                pbar.set_postfix(
                    {"rgb loss" : losses["rgb_loss_psnr"].item()})
                pbar.update()
                if step %config.save_models_every == 0:
                    torch.save(models['ray_bending_estimator'].state_dict(), f"{dir_path}/ray_bending_estimator{step}.pt")
                    torch.save(models['video_downsampler'].state_dict(), f"{dir_path}/video_downsampler_{step}.pt")
                    torch.save(models['spatial_feature_aggregation'].state_dict(), f"{dir_path}/spatial_feature_aggregation_{step}.pt")
                    torch.save(models['spatial_encoder'].state_dict(), f"{dir_path}/spatial_encoder_{step}.pt")
                    torch.save(models['nerf'].state_dict(), f"{dir_path}/nerf_{step}.pt")
                step += 1

        torch.save(models['ray_bending_estimator'].state_dict(), f"{dir_path}/ray_bending_estimator.pt")
        torch.save(models['video_downsampler'].state_dict(), f"{dir_path}/video_downsampler.pt")
        torch.save(models['spatial_feature_aggregation'].state_dict(), f"{dir_path}/spatial_feature_aggregation.pt")
        torch.save(models['spatial_encoder'].state_dict(), f"{dir_path}/spatial_encoder.pt")
        torch.save(models['nerf'].state_dict(), f"{dir_path}/nerf.pt")

        wandb.save(f"{dir_path}/ray_bending_estimator.pt")
        wandb.save(f"{dir_path}/video_downsampler.pt")
        wandb.save(f"{dir_path}/spatial_feature_aggregation.pt")
        wandb.save(f"{dir_path}/spatial_encoder.pt")
        wandb.save(f"{dir_path}/nerf.pt")
    
    run.finish()



def train(run,ray_bender:PointTrajectoryNoODE,
          video_downsampler:VectorEncoder,
          spatial_feature_aggregation: SpatialFeatureAggregation,
          spatial_encoder,
          nerf:NeRF,
          static_field_NeRF:NeRF,
          static_encoder,
          train_loader,
          optimizer_group:OptimizerGroup,
          criterion,
          config,
          device):
    torch.autograd.set_detect_anomaly(True)
    
    # run = {}
    # run["name"] = run
    dir_path = f"{config.checkpoint_dir}/{config.run_name}"
    os.makedirs(dir_path, exist_ok=True)
    
    ray_bender.train()
    video_downsampler.train()
    spatial_feature_aggregation.train()
    nerf.train()
    static_field_NeRF.train()
    static_encoder.train()
    step = 0
    freeze_parameters(nerf)
    freeze_parameters(static_field_NeRF)
    freeze_parameters(static_encoder)
    freeze_parameters(spatial_feature_aggregation)
    freeze_parameters(spatial_encoder)
    freeze_parameters(ray_bender)
    for epoch in range(config.training.epochs):
        if epoch>=config.training.warmup_epochs:
            unfreeze_parameters(nerf)
            unfreeze_parameters(static_field_NeRF)
            unfreeze_parameters(static_encoder)
            unfreeze_parameters(spatial_feature_aggregation)
            unfreeze_parameters(spatial_encoder)
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config.training.epochs}") as pbar:
            for _, (index, rays_o, rays_d, bds, masks, disparity, fwd_flow, fwd_flow_mask, bwd_flow, bwd_flow_mask, target) in enumerate(tqdm(train_loader)):
                    
                loss = 0
                target = target.to(device)
                optimizer_group.zero_grad()
                scene_index, image_indices, y,x = index
                f_st = train_loader.dataset.image_encodings_static[scene_index, image_indices,:,y,x]
                tensors_to_device = [scene_index,image_indices,rays_o, rays_d, bds, f_st, masks, disparity, fwd_flow, fwd_flow_mask, bwd_flow, bwd_flow_mask]
                scene_index,image_indices,rays_o, rays_d, bds, f_st, masks, disparity, fwd_flow, fwd_flow_mask, bwd_flow, bwd_flow_mask = map(lambda x: x.to(device), tensors_to_device)
                masks = masks.unsqueeze(-1)
                
                image_indices = torch.cat(
                    [image_indices-1, image_indices, image_indices+1])
                image_indices = torch.clamp(
                    image_indices, 0, train_loader.dataset.n_images-1)
                scene_indices_three = torch.cat(
                    [scene_index, scene_index, scene_index])
                pose, intrinsics = train_loader.dataset.get_pose_intrinsics(
                    scene_indices_three, image_indices)
                image_indices = image_indices.reshape(3, rays_o.shape[0])
                scene_indices_three = scene_indices_three.reshape(
                    3, rays_o.shape[0])
                pose, intrinsics = pose.reshape(
                    3, rays_o.shape[0], 4, 4), intrinsics.reshape(3, rays_o.shape[0], 4, 4)
                video_embedding = train_loader.dataset.get_video_embedding().to(device)
                f_temp = video_downsampler(video_embedding)
                near, far = bds[...,
                    0].unsqueeze(-1), bds[..., 1].unsqueeze(-1)
                rays_o, rays_d = ndc_rays(train_loader.dataset.h, train_loader.dataset.w,
                    intrinsics[1, :, 0, 0], 1., rays_o, rays_d)
                if (rays_o>100).any():
                    assert False, "rays_o is too big"
                near, far = 0 * torch.ones_like(rays_d[...,:1]), 1 * torch.ones_like(rays_d[...,:1])

                points, z_vals = get_points_along_rays(
                    rays_o, rays_d, near, far, False, config.perturb, config.architecture.n_samples)
                trajectory = models['ray_bending_estimator'](
                    points, f_temp, image_indices[1], scene_index, time_span=config.time_span)

                f_st = models['static_encoder'](f_st)
                f_st = f_st.unsqueeze(1).expand(-1, points.shape[1], -1)
                points_pos_enc = positional_encoding(
                    points.unsqueeze(-1)).reshape(points.shape[0], points.shape[1], -1)
                time = image_indices[1].unsqueeze(1).unsqueeze(2).expand(-1,points.shape[1],-1).unsqueeze(-1)
                time = positional_encoding(time,num_encodings = 5).reshape((points.shape[0], points.shape[1], -1))
                points_encoded_static = torch.cat(
                    [points_pos_enc, f_st], dim=-1)
                outputs_static = models['static_field_NeRF'](points_encoded_static)
                outputs_static_background = outputs_static[..., :-1].clone().detach()
                outputs_static_background[...,:3]=-1e10
                rgb_static, *_ = raw2outputs(
                    outputs_static, z_vals, rays_d, white_bkgd=True)
                rgb_static_masked = rgb_static * (1-masks)
                
                target_static_masked = target * (1-masks)
                if (masks==1).all():
                    rgb_loss_static = 0
                else:
                    rgb_loss_static = criterion(rgb_static_masked, target_static_masked).sum()/((1-masks).sum())
                # else:
                trajectory_shape = trajectory.shape
                pose = pose.unsqueeze(2).expand(-1,-1, config.architecture.n_samples, -1, -1)
                intrinsics = intrinsics.unsqueeze(2).expand(-1,-1, config.architecture.n_samples, -1, -1)
                
                trajectory_2d = project_trajectory_to_image_coords(
                    config, pose, intrinsics, trajectory, trajectory_shape)

                f_sp, pre, cur, post = spatial_feature_aggregation(
                    trajectory_2d, config.image_size[0]//config.downscale, config.image_size[1]//config.downscale, (scene_indices_three, image_indices),train_loader.dataset.image_encodings)

                f_temp = f_temp[scene_index].unsqueeze(1).expand(-1,f_sp.shape[1],-1)

                f_dy, f_dy_pre, f_dy_cur, f_dy_post = torch.cat([f_temp, f_sp], dim=-1),\
                    torch.cat([f_temp, pre], dim=-1),\
                    torch.cat([f_temp, cur], dim=-1),\
                    torch.cat([f_temp, post], dim=-1)

                points_encoded = torch.cat([points_pos_enc, f_dy,time], dim=-1)
                points_encoded_pre = torch.cat(
                    [points_pos_enc, f_dy_pre,time], dim=-1)
                points_encoded_cur = torch.cat(
                    [points_pos_enc, f_dy_cur,time], dim=-1)
                points_encoded_post = torch.cat(
                    [points_pos_enc, f_dy_post,time], dim=-1)
                outputs = nerf(points_encoded)
                outputs_pre = nerf(points_encoded_pre)
                outputs_cur = nerf(points_encoded_cur)
                outputs_post = nerf(points_encoded_post)
                raw_s = outputs_static[..., :4]
                blending = outputs[..., 4]
                rgb_pre,*_ = raw2outputs(outputs_pre,z_vals,rays_d)
                rgb_cur,*_ = raw2outputs(outputs_cur,z_vals,rays_d)
                rgb_post,*_ = raw2outputs(outputs_post,z_vals,rays_d)
                rgb, depth_map_full, _, _, \
                    _, _, _, _, \
                    _, _, _, weights_d, \
                    dynamicness_map = raw2outputs_dynamic(
                        raw_s, outputs[..., :4], blending, z_vals, rays_d)
                masks_denom = masks.sum() 
                if masks_denom == 0:
                    masks_denom = 1
                dynamicness_loss = (torch.abs(dynamicness_map - masks.squeeze())).sum()/masks_denom
                fwd_flow_loss = supervise_flows(trajectory[1],trajectory[2],outputs[...,3],fwd_flow,fwd_flow_mask,pose[2],intrinsics[2],z_vals,rays_d,torch.nn.MSELoss())
                bwd_flow_loss = supervise_flows(trajectory[1],trajectory[0],outputs[...,3],bwd_flow,bwd_flow_mask,pose[0],intrinsics[0],z_vals,rays_d,torch.nn.MSELoss())
                blending_reg = calculate_blending_regularization(
                    blending, z_vals, -disparity, config.depth_epsilon)
                if (masks != 0).any():
                    target_foreground = target*masks
                    rgb_pre = rgb_pre*masks
                    rgb_cur = rgb_cur*masks
                    rgb_post = rgb_post * masks
                    
                    l_bw, l_curr, l_fw = criterion(rgb_pre, target_foreground), criterion(rgb_cur, target_foreground), criterion(rgb_post, target_foreground)
                    l_bw, l_curr, l_fw = l_bw/(masks.sum()), l_curr/(masks.sum()), l_fw/(masks.sum())
                else:
                    l_bw,l_curr,l_fw = torch.tensor(0),torch.tensor(0),torch.tensor(0)
                L_corr = (l_curr + l_bw + l_fw)/3
                rgb_loss = criterion(rgb, target).sum(dim=-1).mean()
                # rgb_dynamic_loss = criterion(rgb_dynamic, target_foreground).sum(dim=-1).mean()
                rgb_loss_psnr = mse2psnr(rgb_loss)
                rgb_loss_static_psnr = mse2psnr(rgb_loss_static)
                rgb_loss_corr_psnr = mse2psnr(L_corr)
                # rgb_loss_dynamic_psnr = mse2psnr(rgb_dynamic_loss)
                rgb_pre_psnr, rgb_cur_psnr, rgb_post_psnr = mse2psnr(l_bw), mse2psnr(l_curr), mse2psnr(l_fw)
                mask_constraint = compute_mask_flow_loss(
                    outputs_pre[..., 4], outputs_cur[..., 4], outputs_post[..., 4])
                disparity_loss = compute_depth_loss(
                    depth_map_full, -disparity)
                blending = torch.sigmoid(blending)
                sparse_loss = entropy(weights_d) + entropy(blending)
                slow_loss = L1_norm(fwd_flow) + L1_norm(bwd_flow)

                loss = rgb_loss*config.loss.rgb_lambda + L_corr*config.loss.correlation_lambda + \
                    rgb_loss_static*config.loss.rgb_static_lambda + disparity_loss*config.loss.disparity_loss_lambda + \
                    blending_reg*config.loss.blending_loss_lambda + mask_constraint*config.loss.mask_constraint_lambda + \
                    sparse_loss*config.loss.sparse_loss_lambda + \
                    (fwd_flow_loss + bwd_flow_loss) * config.loss.nerf_flow_loss_lambda + \
                    slow_loss*config.loss.slow_loss_lambda + \
                        dynamicness_loss*config.loss.dynamicness_lambda
                lrs = optimizer_group.get_lr()
                # loss = fwd_flow_loss + bwd_flow_loss + slow_loss
                wandb.log({
                    "rgb_loss": rgb_loss.item(),
                    "L_corr": L_corr.item(),
                    "rgb_loss_static": rgb_loss_static.item(),
                    "disparity_loss": disparity_loss.item(),
                    "blending_reg": blending_reg.item(),
                    "mask_constraint": mask_constraint.item(),
                    "sparse_loss": sparse_loss.item(),
                    "fwd_flow_loss": fwd_flow_loss.item(),
                    "bwd_flow_loss": bwd_flow_loss.item(),
                    "rgb_loss_psnr": rgb_loss_psnr.item(),
                    "rgb_loss_static_psnr": rgb_loss_static_psnr.item(),
                    "rgb_loss_corr_psnr": rgb_loss_corr_psnr.item(),
                    # "rgb_dynamic_loss": rgb_dynamic_loss.item(),
                    # "rgb_loss_dynamic_psnr": rgb_loss_dynamic_psnr.item(),
                    "dynamicness_loss": dynamicness_loss.item(),
                    "rgb_cur": l_curr.item(),
                    "rgb_pre": l_bw.item(),
                    "rgb_post": l_fw.item(),
                    "rgb_cur_psnr": rgb_cur_psnr.item(),
                    "rgb_pre_psnr": rgb_pre_psnr.item(),
                    "rgb_post_psnr": rgb_post_psnr.item(),
                    "lr_general": lrs[0],
                    "lr_ray_bender": lrs[1],
                    "batch_loss": loss.item()
                })
                pbar.set_postfix(
                    {"fwd_flow_loss" : fwd_flow_loss.item(), "bwd_flow_loss" : bwd_flow_loss.item()})

                loss.backward()
                optimizer_group.step()
                optimizer_group.scheduler_step()
                pbar.update()
                step += 1
        if epoch %2 == 0:
            torch.save(ray_bender.state_dict(), f"{dir_path}/ray_bender_{epoch}.pt")
            torch.save(video_downsampler.state_dict(), f"{dir_path}/video_downsampler_{epoch}.pt")
            torch.save(spatial_feature_aggregation.state_dict(), f"{dir_path}/spatial_feature_aggregation_{epoch}.pt")
            torch.save(spatial_encoder.state_dict(), f"{dir_path}/spatial_encoder_{epoch}.pt")
            torch.save(nerf.state_dict(), f"{dir_path}/nerf_{epoch}.pt")
            # torch.save(static_field_NeRF.state_dict(), f"{dir_path}/static_field_NeRF_{epoch}.pt")
            # torch.save(static_encoder.state_dict(), f"{dir_path}/static_encoder_{epoch}.pt")
            
        torch.save(ray_bender.state_dict(), f"{dir_path}/ray_bender.pt")
        torch.save(video_downsampler.state_dict(), f"{dir_path}/video_downsampler.pt")
        torch.save(spatial_feature_aggregation.state_dict(), f"{dir_path}/spatial_feature_aggregation.pt")
        torch.save(spatial_encoder.state_dict(), f"{dir_path}/spatial_encoder.pt")
        torch.save(nerf.state_dict(), f"{dir_path}/nerf.pt")
        # torch.save(static_field_NeRF.state_dict(), f"{dir_path}/static_field_NeRF.pt")
        # torch.save(static_encoder.state_dict(), f"{dir_path}/static_encoder.pt")

        wandb.save(f"{dir_path}/ray_bender.pt")
        wandb.save(f"{dir_path}/video_downsampler.pt")
        wandb.save(f"{dir_path}/spatial_feature_aggregation.pt")
        wandb.save(f"{dir_path}/spatial_encoder.pt")
        wandb.save(f"{dir_path}/nerf.pt")
        # wandb.save(f"{dir_path}/static_field_NeRF.pt")
        # wandb.save(f"{dir_path}/static_encoder.pt")
    
    run.finish()

def create_dataset_and_networks(config):
    dataset = CustomEncodingsImageDataset(config.data_folders,config,load_images=True)
    networks = {}
    video_embedding = dataset.get_video_embedding()
    networks["video_downsampler"] = VectorEncoder(
        video_embedding.shape[1], [256, 512], 256,normalize_output=False).to(device="cuda")
    networks["spatial_encoder"] = VectorEncoder(256, [256, 512], 256).to(device="cuda")
    networks["ray_bending_estimator"] = PointTrajectoryNoODE(3, 256).to(device="cuda")
    networks["spatial_feature_aggregation"] = SpatialFeatureAggregation(dataset.image_encodings.shape[2],
                                                                      256, dataset.image_encodings).to(device="cuda")
    networks["nerf"] = NeRF(D=config.architecture.layer_count_dynamic,
                          skips=config.architecture.dynamic_layer_skips,input_ch=256*2+60+10)\
                        .to(device="cuda")
    # networks["static_field_NeRF"] = NeRF(D=config.architecture.layer_count_static,skips=config.architecture.static_layer_skips,
        # input_ch=256+60, dynamic=False).to(device="cuda")
    # networks["static_encoder"] = VectorEncoder(512,[256],256,normalize_output=True).to(device="cuda")
    return dataset,networks

def load_checkpoints(model_dict,checkpoint_dir):
    for model_name in model_dict:
        model_dict[model_name].load_state_dict(torch.load(os.path.join(checkpoint_dir,f"{model_name}.pt")))
    return model_dict

def project_trajectory_to_image_coords(config, pose, intrinsics, trajectory, trajectory_shape):
    # trajectory = rearrange(
        # trajectory, 'time b n_samples xyz -> (time b) n_samples xyz')]
    trajectory_world = NDC2world(trajectory,config.image_size[0]//config.downscale,config.image_size[1]//config.downscale, intrinsics[...,0,0].unsqueeze(-1))
    trajectory_2d = project_3d_to_2d(trajectory_world, pose,intrinsics)
    trajectory_2d = rearrange(trajectory_2d, '(time b n_samples) xy -> time b n_samples xy',
                           b=trajectory_shape[1], n_samples=config.architecture.n_samples)
    return trajectory_2d

def main():
    config = get_config()
    dataset = CustomEncodingsImageDataset(config.data_folders,config)

    run = wandb.init(project="mononerf", config=config.to_dict())  
    if hasattr(config,"run_name"):
        run.name = config.run_name
    else:
        run.name = 'mononerf-{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        config.run_name = run.name
    dataset,models = create_dataset_and_networks(config)
    # models = load_checkpoints(models,os.path.join(config.checkpoint_dir,"mononerf-2023-07-12_00-24-33"))
    # models['ray_bending_estimator'].load_state_dict(torch.load("/home/yiftach/main/Research/MonoNeRF/checkpoints/mononerf-2023-07-12_11-25-56/ray_bending_estimator6000.pt"))
    # checkpoint_dir = "/home/yiftach/main/Research/MonoNeRF/checkpoints/dutiful-frog-612"
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size,num_workers=config.num_workers, shuffle=True)
    
    grad_params = list(models['nerf'].parameters())
    grad_params += list(models['spatial_feature_aggregation'].parameters())
    grad_params += list(models['spatial_encoder'].parameters())
    grad_params = list(models['video_downsampler'].parameters())
    
    total_params = sum(p.numel() for p in grad_params)+sum(p.numel() for p in models['ray_bending_estimator'].parameters())
    formatted_params = format_number(total_params)

    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("                      NETWORK PARAMETER COUNT                  ")
    print("---------------------------------------------------------------")
    print("      Total number of parameters: {:^20}     ".format(
        formatted_params))
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    part_percentages = []
    for part in ['video_downsampler', 'spatial_encoder', 'ray_bending_estimator', 'spatial_feature_aggregation', 'nerf']:
        part_params = sum(p.numel() for p in models[part].parameters())
        part_percentage = (part_params / total_params) * 100
        part_percentages.append((part, part_percentage))

    part_percentages.sort(key=lambda x: x[1], reverse=True)

    print("Percentage of parameters per part of the network:")
    for part, percentage in part_percentages:
        print("{:<25} {:>10.2f}%".format(part, percentage))

    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    for param in grad_params:
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)

    optimizer_general = torch.optim.Adam([
        {'params': grad_params, 'lr': config.training.lr},
    ])
    optimizer_ray_bender = torch.optim.Adam([
        {'params': models['ray_bending_estimator'].parameters(), 'lr': config.training.ray_bender_lr},
    ])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_ray_bender,gamma=config.training.lr_gamma,milestones=config.training.lr_milestones)
    opt_group = OptimizerGroup()
    opt_group.add(optimizer_general)
    opt_group.add(optimizer_ray_bender,scheduler)
    if config.network_training_function in globals() and callable(globals()[config.network_training_function]):
        eval(config.network_training_function)(run,models, dataloader, opt_group, torch.nn.MSELoss(), config, "cuda")
    else:
        raise Exception("Function {} not found in module {}".format(config.network_training_function,__name__))
    # if config.training_mode=="static+dynamic":
        # train(ray_bending_estimator, video_downsampler, spa, spatial_encoder, nerf, static_field_NeRF,
            # static_encoder, dataloader, opt_group, torch.nn.MSELoss(reduction="none"), config, "cuda")
    # elif config.training_mode=="dynamic":
        # train_dynamic(run,models, dataloader, opt_group, torch.nn.MSELoss(), config, "cuda")


def format_number(number):
    units = ["", "K", "M", "B"]
    unit_index = 0
    while number >= 1000 and unit_index < len(units) - 1:
        number /= 1000
        unit_index += 1
    formatted_number = "{:,.2f}{}".format(number, units[unit_index])
    return formatted_number


if __name__ == '__main__':
    main()
