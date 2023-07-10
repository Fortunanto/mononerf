from util.utils_3d import compute_2d_displacements
from einops import reduce,rearrange,repeat
import torch
from util.ray_helpers import *

def supervise_flows(points_start,points_end,sigma,velocities_2d_gt,mask,pose,intrinsics,z_vals,rays_d,criterion):
    
    n_samples = points_start.shape[1]
    # intrinsics = repeat(intrinsics,"batch a b -> (batch n_samples) a b",n_samples=n_samples)
    # pose = repeat(pose,"batch a b -> (batch n_samples) a b",n_samples=n_samples)
    # points_start = rearrange(points_start,"batch n_samples c -> (batch n_samples) c",n_samples=n_samples)
    # points_end = rearrange(points_end,"batch n_samples c -> (batch n_samples) c",n_samples=n_samples)
    points_start = NDC2world(points_start,540,960,intrinsics[...,0,0].unsqueeze(-1))
    points_end = NDC2world(points_end,540,960,intrinsics[...,0,0].unsqueeze(-1))
    velocities_2d = compute_2d_displacements(points_start,points_end,intrinsics,pose)
    mask = mask.unsqueeze(-1)
    # assert False, f"disp_3d.shape = {disp_3d.shape}"
    velocities_2d = rearrange(velocities_2d,"(batch n_samples) c -> batch n_samples c",n_samples=n_samples)

    velocities_2d = torch.cat([velocities_2d, sigma.unsqueeze(-1)], dim=-1)
    velocities_2d = velocity2outputs(velocities_2d, z_vals, rays_d)
    # velocities_2d = reduce(velocities_2d,"batch n_samples c -> batch c", "mean")

    # velocities_2d_gt = torch.zeros_like(velocities_2d)
    # assert False, f"pose.shape = {pose.shape} intrinsics.shape = {intrinsics.shape} velocities.shape = {velocities.shape} points.shape = {points.shape}"
    velocities_2d = velocities_2d*mask
    velocities_2d_gt = velocities_2d_gt*mask
    # print(velocities_2d[:2])
    # assert not velocities_2d.isnan().any(), "velocities_2d has NaNs, {}".format(velocities_2d)
    return criterion(velocities_2d,velocities_2d_gt)
