from einops import reduce,rearrange,repeat
import torch
from util.ray_helpers import *

def supervise_flows(points_start_2d,points_end_2d,sigma,velocities_2d_gt,mask,pose,intrinsics,z_vals,rays_d,criterion,forward_facing_scene=True):
    
    n_samples = points_start_2d.shape[1]
    mask = mask.unsqueeze(-1)
    velocities_2d = points_end_2d-points_start_2d
    velocities_2d = torch.cat([velocities_2d, sigma.unsqueeze(-1)], dim=-1)
    velocities_2d = velocity2outputs(velocities_2d, z_vals, rays_d)

    velocities_2d = velocities_2d*mask
    velocities_2d_gt = velocities_2d_gt*mask
    return criterion(velocities_2d,velocities_2d_gt)
