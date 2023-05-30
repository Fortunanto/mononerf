from util.utils_3d import compute_2d_displacements
from einops import reduce,rearrange,repeat
import torch

def supervise_flows(points,velocities,velocities_2d_gt,pose,intrinsics,criterion):
    
    n_samples = points.shape[1]
    intrinsics = repeat(intrinsics,"batch a b -> (batch n_samples) a b",n_samples=n_samples)
    pose = repeat(pose,"batch a b -> (batch n_samples) a b",n_samples=n_samples)
    velocities_2d = compute_2d_displacements(points,velocities,intrinsics,pose)

    # assert False, f"disp_3d.shape = {disp_3d.shape}"
    velocities_2d = rearrange(velocities_2d,"(batch n_samples) c -> batch n_samples c",n_samples=n_samples)
    velocities_2d = reduce(velocities_2d,"batch n_samples c -> batch c",reduction="mean")
    # velocities_2d_gt = torch.zeros_like(velocities_2d)
    # assert False, f"pose.shape = {pose.shape} intrinsics.shape = {intrinsics.shape} velocities.shape = {velocities.shape} points.shape = {points.shape}"

    # print(velocities_2d[:2])
    return torch.sqrt(criterion(velocities_2d,velocities_2d_gt))
