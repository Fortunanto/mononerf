from util.utils_3d import project_3d_to_2d_batch,compute_2d_displacements
from einops import reduce,rearrange,repeat
import torch

def supervise_flows(points,velocities_2d_gt,pose,intrinsics,criterion):
    n_samples = points.shape[1]
    intrinsics = repeat(intrinsics,"batch a b -> (batch n_samples) a b",n_samples=n_samples)
    pose = repeat(pose,"batch a b -> (batch n_samples) a b",n_samples=n_samples)
    velocities_2d = rearrange(compute_2d_displacements(points,intrinsics,pose),"(batch n_samples) c -> batch n_samples c",n_samples=n_samples)
    velocities_2d = reduce(velocities_2d,"batch n_samples c -> batch c",reduction="mean")
    return torch.sqrt(criterion(velocities_2d,velocities_2d_gt))
