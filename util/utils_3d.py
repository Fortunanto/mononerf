import torch
from einops import *

import numpy as np


def project_3d_to_2d_batch(points_3d, intrinsic_matrix, pose_matrix):
    """
    Project a batch of 3D points from world coordinates to 2D image coordinates.

    :param points_3d: Batch of 3D points in world coordinates as a tensor of shape [B, 3].
    :param intrinsic_matrix: Batch of camera intrinsic matrices as a tensor of shape [B, 3, 3].
    :param pose_matrix: Batch of camera pose (extrinsic) matrices as a tensor of shape [B, 3, 4].
    :return: Batch of 2D points in image coordinates as a tensor of shape [B, 2], rounded to nearest integers.
    """
    # assert False, f"points_3d.shape: {points_3d.shape} intrinsic_matrix.shape: {intrinsic_matrix.shape} pose_matrix.shape: {pose_matrix.shape}"
    B = points_3d.shape[0]*points_3d.shape[1]
    points_3d = rearrange(points_3d, 'b n_samples xyz -> (b n_samples) xyz')
    # Extend points_3d to homogeneous coordinates
    points_3d_homogeneous = torch.cat([points_3d, torch.ones(B, 1, device=points_3d.device)], dim=-1)
    # Apply extrinsic matrix
    points_cam = torch.bmm(pose_matrix, points_3d_homogeneous.unsqueeze(-1)).squeeze(-1)
    
    # points_cam = 
    # Apply intrinsic matrix
    points_2d_homogeneous = torch.bmm(intrinsic_matrix, points_cam.unsqueeze(-1)).squeeze(-1)

    # Normalize to get actual 2D coordinates
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    # Round the coordinates to nearest integers
    points_2d_rounded = torch.round(points_2d)

    # Swap coordinates to match (row, column) format and return
    return torch.stack([points_2d_rounded[:, 1], points_2d_rounded[:, 0]], dim=-1)

def compute_2d_displacements(points_3d, intrinsic_matrices,pose_matrices, time_step=0.006666667):
    # Compute 3D positions in the next frame
    next_points_3d = points_3d[:, :, 2]
    points_3d = points_3d[:, :, 1]
    # Project 3D points to 2D
    points_2d = project_3d_to_2d_batch(points_3d, intrinsic_matrices,pose_matrices)
    next_points_2d = project_3d_to_2d_batch(next_points_3d, intrinsic_matrices,pose_matrices)
    # Compute 2D displacements
    displacements_2d = next_points_2d - points_2d
    return displacements_2d

if __name__ == "__main__":
    point = torch.randint(200,400,size=(10,2)).to(device="cuda")
    pose = torch.load("data/ball/data/pose.pt").to(device="cuda")
    intrinsic = torch.load("data/ball/data/intrinsics.pt").to(device="cuda")
