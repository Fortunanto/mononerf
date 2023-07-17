import torch
from einops import rearrange
import numpy as np
from tqdm import tqdm 
from util.ray_helpers import get_points_along_rays,project_3d_to_image_coords
from util import positional_encoding
import torch.nn.functional as F
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=1, white_bkgd=False, pytest=False,**kwargs):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    s = torch.Tensor([1e10]).to(dists.device)
    dists = torch.cat([dists, s.expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
        noise = noise.to(dists.device)
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    blending = raw[...,4].mean(-1).unsqueeze(-1)  # [N_rays, 1]
    
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    alpha_ones = torch.ones((alpha.shape[0], 1)).to(alpha.device)
    weights = alpha * torch.cumprod(torch.cat([alpha_ones, 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map,depth_map,weights,blending, disp_map, acc_map

def raw2outputs_dynamic(raw_s,
                raw_d,
                blending,
                z_vals,
                rays_d,
                raw_noise_std=1):
    """Transforms model's predictions to semantically meaningful values.

    Args:
      raw_s: [num_rays, num_samples along ray, 4]. Prediction from Static model.
      raw_d: [num_rays, num_samples along ray, 4]. Prediction from Dynamic model.
      z_vals: [num_rays, num_samples along ray]. Integration time.
      rays_d: [num_rays, 3]. Direction of each ray.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. Inverse of depth map.
      acc_map: [num_rays]. Sum of weights along each ray.
      weights: [num_rays, num_samples]. Weights assigned to each sampled color.
      depth_map: [num_rays]. Estimated distance to object.
    """
    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    def raw2alpha(raw, dists, act_fn=F.relu): return 1.0 - \
        torch.exp(-act_fn(raw) * dists)

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # assert False, f"dists.shape = {dists.shape} device = {dists.device}"
    # The 'distance' from the last integration time is infinity.
    dists = torch.cat(
        [dists, torch.tensor([1e10],device=dists.device).expand(dists[..., :1].shape)],
         -1) # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Extract RGB of each sample position along each ray.
    rgb_d = torch.sigmoid(raw_d[..., :3])  # [N_rays, N_samples, 3]
    rgb_s = torch.sigmoid(raw_s[..., :3])  # [N_rays, N_samples, 3]

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_d[..., 3].shape) * raw_noise_std
        noise = noise.to(dists.device)
    blending = blending
    blending = torch.sigmoid(blending)
    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha_d = raw2alpha(raw_d[..., 3] + noise, dists) # [N_rays, N_samples]
    alpha_s = raw2alpha(raw_s[..., 3] + noise, dists) # [N_rays, N_samples]
    alphas  = 1. - (1. - alpha_s) * (1. - alpha_d) # [N_rays, N_samples]

    T_d    = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1),device=dists.device), 1. - alpha_d + 1e-10], -1), -1)[:, :-1]
    T_s    = torch.cumprod(torch.cat([torch.ones((alpha_s.shape[0], 1),device=dists.device), 1. - alpha_s + 1e-10], -1), -1)[:, :-1]
    T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1),device=dists.device), (1. - alpha_d * blending) * (1. - alpha_s * (1. - blending)) + 1e-10], -1), -1)[:, :-1]
    # T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), torch.pow(1. - alpha_d + 1e-10, blending) * torch.pow(1. - alpha_s + 1e-10, 1. - blending)], -1), -1)[:, :-1]
    # T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), (1. - alpha_d) * (1. - alpha_s) + 1e-10], -1), -1)[:, :-1]

    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    weights_d = alpha_d * T_d
    weights_s = alpha_s * T_s
    weights_full = (alpha_d * blending + alpha_s * (1. - blending)) * T_full
    # weights_full = alphas * T_full

    # Computed weighted color of each sample along each ray.
    rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)
    rgb_map_s = torch.sum(weights_s[..., None] * rgb_s, -2)
    rgb_map_full = torch.sum(
        (T_full * alpha_d * blending)[..., None] * rgb_d + \
        (T_full * alpha_s * (1. - blending))[..., None] * rgb_s, -2)

    # Estimated depth map is expected distance.
    depth_map_d = torch.sum(weights_d * z_vals, -1)
    depth_map_s = torch.sum(weights_s * z_vals, -1)
    depth_map_full = torch.sum(weights_full * z_vals, -1)

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map_d = torch.sum(weights_d, -1)
    acc_map_s = torch.sum(weights_s, -1)
    acc_map_full = torch.sum(weights_full, -1)

    # Computed dynamicness
    dynamicness_map = torch.sum(weights_full * blending, -1)
    # dynamicness_map = 1 - T_d[..., -1]
    
    return rgb_map_full, depth_map_full, acc_map_full, weights_full, \
           rgb_map_s, depth_map_s, acc_map_s, weights_s, \
           rgb_map_d, depth_map_d, acc_map_d, weights_d, dynamicness_map


def batchify_rays(train_loader, s, t, models, video_embedding, config, chunk=1024, verbose=True,device="cuda", **kwargs):
    rays_o_render, rays_d_render = train_loader.dataset.rays_o_np[s,t],train_loader.dataset.rays_d_np[s,t]
    rays_o_render = rays_o_render.transpose(1,2,0)
    rays_d_render = rays_d_render.transpose(1,2,0)

    poses_render = train_loader.dataset.poses[s,[t-1,t,t+1]].to(device)
    intrinsics_render = train_loader.dataset.intrinsics[s,[t-1,t,t+1]].to(device)
    
    rays_o_flat = rearrange(rays_o_render,'h w c -> (h w) c')
    rays_d_flat = rearrange(rays_d_render,'h w c -> (h w) c')
    rays_o_flat = torch.from_numpy(rays_o_flat).to(device="cuda")
    rays_d_flat = torch.from_numpy(rays_d_flat).to(device="cuda")
    
    result = {}
    result["rgb_full"] = torch.zeros(*rays_o_flat.shape).to(device="cuda")
    
    with torch.no_grad():
        f_temp = models['video_downsampler'](video_embedding)
        for i in tqdm(range(0, result["rgb_full"].shape[0], chunk),disable=not verbose):
            size = min(chunk,result["rgb_full"].shape[0]-i)
            pose_batch = poses_render.unsqueeze(0).expand(size,-1,-1,-1)
            intrinsics_batch = intrinsics_render.unsqueeze(0).expand(size,-1,-1,-1)
            scene_indices = (torch.ones(size,dtype=torch.int64)*s).to(device="cuda")
            image_indices = (torch.ones(size,dtype=torch.int64)*t).unsqueeze(-1).to(device="cuda")
            image_indices = torch.cat([image_indices-1,image_indices,image_indices+1],dim=-1)
            
            res=render_rays(rays_o_flat[i:i+chunk],
                                rays_d_flat[i:i+chunk],
                                models,f_temp,pose_batch,intrinsics_batch,
                                image_indices=image_indices,scene_indices=scene_indices,config=config,
                                all_parts=False,
                                raw_noise_std=0,white_bkgd=True,
                                **kwargs)["rgb_full"]
            result["rgb_full"][i:i+chunk]=res
    return result


def render_rays(rays_o,rays_d,networks,f_temp,pose,intrinsics,image_indices,scene_indices,config,all_parts=False,**kwargs):
    result = {}
    h,w = 360,340
    near, far = 0 * torch.ones_like(rays_d[:,:1]), 1 * torch.ones_like(rays_d[:,:1])
    points, z_vals = get_points_along_rays(
        rays_o, rays_d, near, far, False, **kwargs)
    trajectory = networks['ray_bending_estimator'](
        points, f_temp, image_indices[:,1], scene_indices, 3)
    result["trajectory"] = trajectory
    pose_batch = pose.unsqueeze(2).expand(-1,-1, 64, -1, -1)
    intrinsics_batch = intrinsics.unsqueeze(2).expand(-1,-1, 64, -1, -1)
    trajectory_2d = project_3d_to_image_coords(
        h,w, pose_batch, intrinsics_batch, trajectory.float(),**kwargs) 
    result["trajectory_2d"] = trajectory_2d
    points_pos_enc = positional_encoding(
        points.unsqueeze(-1)).reshape(points.shape[0], points.shape[1], -1)
    f_sp, f_sp_pre,f_sp_cur,f_sp_post = networks['spatial_feature_aggregation'](
            trajectory_2d, h, w, (scene_indices.unsqueeze(0).expand(3,*scene_indices.shape),image_indices)) 
    f_temp_cur = f_temp[scene_indices].unsqueeze(1).expand(-1,f_sp.shape[1],-1)
    f_dy = torch.cat([f_temp_cur, f_sp], dim=-1)
    time = positional_encoding(image_indices[:,1].unsqueeze(-1),num_encodings = 5).unsqueeze(1).expand(-1,points_pos_enc.shape[1],-1)
    points_encoded = torch.cat([points_pos_enc, f_dy,time], dim=-1)
    outputs = networks['nerf'](points_encoded)
    result['rgb_full'], result['depth_map_full'],\
        result["weights_full"],result["blending_full"],\
        result["displacement_map_full"],result["accuracy_map_full"] = raw2outputs(outputs, z_vals.float(), rays_d.float(),**kwargs)
    result['output_full'] = outputs
    result['z_vals'] = z_vals
    if all_parts:
        f_dy_pre, f_dy_cur, f_dy_post = torch.cat([f_temp_cur,f_sp_pre],dim=-1), torch.cat([f_temp_cur,f_sp_cur],dim=-1), torch.cat([f_temp_cur,f_sp_post],dim=-1)
        points_encoded_pre, points_encoded_cur, points_encoded_post = torch.cat([points_pos_enc, f_dy_pre,time], dim=-1), torch.cat([points_pos_enc, f_dy_cur,time], dim=-1), torch.cat([points_pos_enc, f_dy_post,time], dim=-1)
        outputs_pre, outputs_cur, outputs_post = networks['nerf'](points_encoded_pre),\
                                                    networks['nerf'](points_encoded_cur),\
                                                        networks['nerf'](points_encoded_post)
        result['output_pre'] = outputs_pre
        result['output_cur'] = outputs_cur
        result['output_post'] = outputs_post        
        result['rgb_pre'], result['depth_map_pre'],\
        result["weights_pre"],result["blending_pre"],\
        result["displacement_map_pre"],result["accuracy_map_pre"] = raw2outputs(outputs_pre, z_vals.float(), rays_d.float(),**kwargs)
        result['rgb_cur'], result['depth_map_cur'],\
        result["weights_cur"],result["blending_cur"],\
        result["displacement_map_cur"],result["accuracy_map_cur"] = raw2outputs(outputs_cur, z_vals.float(), rays_d.float(),**kwargs)
        result['rgb_post'], result['depth_map_post'],\
        result["weights_post"],result["blending_post"],\
        result["displacement_map_post"],result["accuracy_map_post"] = raw2outputs(outputs_post, z_vals.float(), rays_d.float(),**kwargs)
        
    return result
