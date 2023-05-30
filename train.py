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
# from util.util import group_indices
from einops import *
from loss import *
import wandb
import copy
import torch
import torch.autograd as autograd
from torchviz import make_dot
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

def train(ray_bender,video_downsampler,spatial_feature_aggregation,spatial_encoder,nerf, train_loader, optimizer, criterion,config, device):
    # flow_loss = 0
    epoch_flow_loss = 0
    num_batches = 0

    prev_parameters = None
    # ray_bender.train()
    criterion = torch.nn.MSELoss()
    best_epoch_flow_loss = float('inf')  # Initialize with very high value
    checkpoint_dir = "./checkpoints"  # You may want to change this
    os.makedirs(checkpoint_dir, exist_ok=True)
    prev_parameters = None
    for epoch in range(config.epochs):
        ray_bender.train()
        prev_parameters = copy.deepcopy(ray_bender.state_dict())

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}") as pbar:
            for batch_idx, (index,rays_o,rays_d, target) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                
                image_indices,x,y = index
                rays_o = rays_o.to(device)
                rays_d = rays_d.to(device)
                image_indices = image_indices.to(device)
                image_indices = torch.cat([image_indices-1,image_indices,image_indices+1])
                # image_indices_grouped = group_indices(image_indices)

                # assert False, f"image_indices.shape: {image_indices.shape}"
                pose,intrinsics = train_loader.dataset.get_pose_intrinsics(image_indices)
                image_indices = image_indices.reshape(3,rays_o.shape[0])
                pose,intrinsics = pose.reshape(3,rays_o.shape[0],4,4),intrinsics.reshape(3,rays_o.shape[0],4,4)
                flows = train_loader.dataset.get_video_flows()
                video_embedding = train_loader.dataset.get_video_embedding()
                f_temp = video_downsampler(video_embedding)
                points = get_points_along_rays(rays_o,rays_d,config.near,config.far,config.linear_displacement,config.perturb,config.n_samples)
                
                trajectory,velocity = ray_bender(points,f_temp,intrinsics[1],pose[1],image_indices[1],time_span=config.time_span)

                flow_loss = supervise_flows(trajectory[1],velocity,flows[image_indices[1],:,x,y],pose[1],intrinsics[1],criterion)

                trajectory_shape = trajectory.shape

                pose = repeat(pose,'time b x y -> (time b n_samples) x y',x=4,y=4,n_samples=config.n_samples)
                intrinsics = repeat(intrinsics,'time b x y -> (time b n_samples) x y',x=4,y=4,n_samples=config.n_samples)
                trajectory = rearrange(trajectory,'time b n_samples xyz -> (time b) n_samples xyz')
                trajectory_2d = project_3d_to_2d_batch(trajectory,pose,intrinsics)
                trajectory_2d = rearrange(trajectory_2d,'(time b n_samples)  xy -> time b n_samples xy',b=trajectory_shape[1],n_samples=config.n_samples)
                trajectory_2d = reduce(trajectory_2d,'time b n_samples xy -> time b xy',reduction='mean')
                trajectory_2d = adjust_to_image_coordinates(trajectory_2d,config.image_size)
                feat = spatial_feature_aggregation(trajectory_2d,config.image_size[0],config.image_size[1],image_indices)
                feat_loss = criterion(feat,torch.ones_like(feat)+1)
                # assert False, f"check_full_differentiability(feat_loss) {check_differentiability(feat_loss)}"
                feat_loss.backward()
                # assert False, f"feat.grad_fn: {feat.grad_fn}"
                # f_sp = spatial_encoder(feat)
                # f_temp = f_temp.expand(f_sp.shape[0],-1)
                # f_dy = torch.cat([f_temp,f_sp],dim=1)
                
                # assert False, f"f_temp shape {f_temp.shape} f_sp shape {f_sp.shape} f_dy shape {f_dy.shape}"
                # assert False, f"aggregated_features_trajectory.shape: {aggregated_features_trajectory.shape}"
                # flow_loss.backward()
                optimizer.step()
                # wandb.log({"flow_loss Train": flow_loss.item()})

                epoch_flow_loss += flow_loss.item()
                num_batches += 1
                pbar.set_postfix({"Feat Loss Train": feat_loss.item()})
                pbar.update()


            # Update previous parameters
            epoch_flow_loss /= num_batches
            # wandb.log({"Epoch Flow Loss Train": epoch_flow_loss.item()})

            # if epoch_flow_loss < best_epoch_flow_loss:
            #     best_epoch_flow_loss = epoch_flow_loss

            #     # Save both ray_bender and video_downsampler models
            #     torch.save(ray_bender.state_dict(), os.path.join(checkpoint_dir, 'best_ray_bender_model.pth'))
            #     torch.save(video_downsampler.state_dict(), os.path.join(checkpoint_dir, 'best_video_downsampler_model.pth'))

            # wandb.log({"epoch_flow_loss": epoch_flow_loss})
            epoch_flow_loss = 0
            num_batches = 0
            parameters_changed = check_parameters_changed(ray_bender, prev_parameters)
            print(f"Parameters changed: {parameters_changed}")

        

def main():
    config = get_config()
    # encoded_data_path = os.path.join(config.data_folder,"data")
    dataset = CustomEncodingsImageDataset(config.data_folder)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # video_embedding_path = os.path.join(encoded_data_path,"video_embedding.pt")
    video_embedding = dataloader.dataset.get_video_embedding()
    video_downsampler = VectorEncoder(video_embedding.shape[0], [2048,2048, 2048, 2048],256).to(device="cuda")
    spatial_encoder = VectorEncoder(256,0,256).to(device="cuda")
    ray_bending_estimator = PointTrajectory(3,256).to(device="cuda")
    # assert False, f"dataset.image_encodings {dataset.image_encodings.shape} video_embedding {video_embedding.shape}"
    spa = SpatialFeatureAggregation(256,dataset.image_encodings).to(device="cuda")
    for param in ray_bending_estimator.parameters():
        # print(f"param {param}")
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    for param in video_downsampler.parameters():
            print(f"param {param}")
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    # wandb.init(project=config.exp_group_name, config=config)
    # wandb.watch(ray_bending_estimator)
    # wandb.watch(video_downsampler)
    optimizer = torch.optim.Adam(list(ray_bending_estimator.parameters())+list(video_downsampler.parameters())+list(spatial_encoder.parameters())+list(spa.parameters()), lr=config.lr)
    train(ray_bending_estimator,video_downsampler,spa,spatial_encoder,None,dataloader,optimizer,None,config,"cuda")
    pass
    
if __name__ == '__main__':
    main()
