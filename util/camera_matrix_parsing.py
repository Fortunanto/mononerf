import numpy as np
import torch

cameras = np.load("/home/yiftach/main/Research/MonoNeRF/data/ball/cameras_sphere.npz")
# assert False, f"cameras.files: {cameras.files}"
# world_mat is a projection matrix from world to image
df_factor = 3/5
downsample_matrix = np.array([[df_factor, 0, 0, 0],[0, df_factor, 0, 0],[0, 0, df_factor, 0],[0, 0, 0, 1]])
camera_mats_np = [(cameras['camera_mat_%d' % idx].astype(np.float32)@downsample_matrix)[np.newaxis,:,:] for idx in range(151)]
camera_mats_np = np.concatenate(camera_mats_np, axis=0)
camera_mats = torch.from_numpy(camera_mats_np).float().to("cuda")   # [n_images, 4, 4]
torch.save(camera_mats, "data/ball/data/c2w.pt")
assert False, f"camera_mats {camera_mats.shape}"
self.scale_mats_np = []

# scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
self.scale_mats_np = [self.cameras['scale_mat_%d' % idx].astype(np.float32)@downsample_matrix for idx in range(self.n_images)]

self.intrinsics_all = []
self.pose_all = []

for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
    P = world_mat @ scale_mat
    P = P[:3, :4]
    intrinsics,pose = load_K_Rt_from_P(None, P)
    self.pose_all.append(torch.from_numpy(pose).float())
    self.intrinsics_all.append(torch.from_numpy(intrinsics).float())

self.pose_all = torch.stack(self.pose_all).to(self.device)   # [n_images, 3, 3]
self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 3, 3]