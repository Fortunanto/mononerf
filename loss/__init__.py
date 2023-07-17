from einops import reduce,rearrange,repeat
import torch
from util.ray_helpers import *
from util.utils_3d import compute_2d_displacements
def supervise_flows(points_start,points_end,sigma,velocities_2d_gt,mask,pose,intrinsics,z_vals,rays_d,criterion,forward_facing_scene=True):
    h = int(intrinsics[0,0,2]*2)
    w = int(intrinsics[0,1,2]*2)
    f = intrinsics[0,0,0]
    if forward_facing_scene:
        points_start = NDC2world(points_start,h,w, f)
        points_end = NDC2world(points_end,h,w, f)
    # points_start = NDC2world(points_start,)
    n_samples = points_start.shape[1]
    intrinsics = repeat(intrinsics,"batch a b -> batch n_samples a b",n_samples=n_samples)
    pose = repeat(pose,"batch a b -> batch n_samples a b",n_samples=n_samples)
    velocities_2d = compute_2d_displacements(points_start,points_end,intrinsics,pose,forward_facing_scene)
    # mask = mask
    if torch.isnan(velocities_2d).any():
        print("really fuck off")
    velocities_2d = rearrange(velocities_2d,"(batch n_samples) a -> batch n_samples a",n_samples=n_samples)
    if torch.isnan(velocities_2d).any():
        print("really fuck off 2")

    velocities_2d = torch.cat([velocities_2d, sigma.unsqueeze(-1)], dim=-1)
    if torch.isnan(velocities_2d).any():
        print("really fuck off 3")

    velocities_2d = velocity2outputs(velocities_2d, z_vals, rays_d)
    if torch.isnan(velocities_2d).any():
        print("really fuck off 4")
    velocities_2d[(1-mask).bool().squeeze()]=0

    if torch.isnan(velocities_2d).any():
        print("really fuck off 5")
 
    velocities_2d_gt = velocities_2d_gt*mask
    if torch.isnan(criterion(velocities_2d,velocities_2d_gt)).any():
        print("fuck off")        
    return criterion(velocities_2d,velocities_2d_gt)
