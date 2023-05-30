from util.ray_helpers import get_rays_np
import torch
import numpy as np

intrinsics = torch.load("data/ball/data/intrinsics.pt").to(device="cuda:0")
intrinsics = intrinsics.cpu().numpy()
pose = torch.load("data/ball/data/pose.pt").to(device="cuda:0")
pose = pose.cpu().numpy()
rays_o_t, rays_d_t = [],[]
for i in range(intrinsics.shape[0]):
    rays_o,rays_d = get_rays_np(240,360,intrinsics[i],pose[i])
    rays_o_t.append(rays_o)
    rays_d_t.append(rays_d)
rays_o_t = np.stack(rays_o_t)
rays_d_t = np.stack(rays_d_t)
np.save("data/ball/data/rays_o.npy",rays_o_t)
np.save("data/ball/data/rays_d.npy",rays_d_t)
assert False, f"rays_o_t.shape = {rays_o_t.shape} rays_d_t.shape = {rays_d_t.shape}"