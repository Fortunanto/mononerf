import torch
from einops import *

import numpy as np

def adjust_to_image_coordinates(points_2d, img_size):
    points_2d[..., 1] += img_size[1] / 2  # Add half the width to the x-coordinate
    points_2d[..., 0] = img_size[0] / 2 - points_2d[..., 0]  # Subtract y-coordinate from half the height
    return points_2d

def interpolate_feature_map(points,h,w,indices,features):
    interp,mask = get_interpolation_indices(points=points, h=h, w=w)
    mask = rearrange(mask,'time b -> (time b)')
    interp = rearrange(interp.to(features.device),'time b bounds xy -> (time b) bounds xy')
    # assert False, f"interp shape: {interp.shape}"
    points = rearrange(points,'time b xy -> (time b) xy')
    indices = indices.reshape(-1).to(features.device)
    # assert False, f"indices shape: {indices.shape} interp shape: {interp.shape}"
    x0y0 = interp[:,0]
    x0y1 = interp[:,1]
    x1y0 = interp[:,2]
    x1y1 = interp[:,3]
    
    features_topleft = features[indices,:,x0y0[:,0],x0y0[:,1]].to(points.device)
    features_topright = features[indices,:,x0y1[:,0],x0y1[:,1]].to(points.device)
    features_bottomleft = features[indices,:,x1y0[:,0],x1y0[:,1]].to(points.device)
    features_bottomright = features[indices,:,x1y1[:,0],x1y1[:,1]].to(points.device)
    x0y0 = x0y0.to(points.device)
    x0y1 = x0y1.to(points.device)
    x1y0 = x1y0.to(points.device)
    x1y1 = x1y1.to(points.device)

    wa = ((points[:,0]-x0y0[:,0])*(points[:,1]-x0y0[:,1])).unsqueeze(1)
    wb = ((points[:,0]-x0y1[:,0])*(x0y1[:,1]-points[:,1])).unsqueeze(1)
    wc = ((x1y0[:,0]-points[:,0])*(points[:,1]-x1y0[:,1])).unsqueeze(1)
    wd = ((x1y1[:,0]-points[:,0])*(x1y1[:,1]-points[:,1])).unsqueeze(1)
    interp_features = wa * features_topleft + wb * features_topright + wc * features_bottomleft + wd * features_bottomright
    return mask.unsqueeze(1)*interp_features

def get_interpolation_indices(points,h,w):
    x0 = torch.floor(points[...,0]).long()
    x1 = x0 + 1
    y0 = torch.floor(points[...,1]).long()
    y1 = y0 + 1

    mask = (x0 < 0) | (x0 >= 100) | (y0 < 0) | (y0 >= 100)

    x0 = x0*(~mask)
    x1 = x1*(~mask)
    y0 = y0*(~mask)
    y1 = y1*(~mask)    
       
    x0y0 = torch.stack((x0, y0), dim=-1).unsqueeze(2)
    x0y1 = torch.stack((x0, y1), dim=-1).unsqueeze(2)
    x1y0 = torch.stack((x1, y0), dim=-1).unsqueeze(2)
    x1y1 = torch.stack((x1, y1), dim=-1).unsqueeze(2)
    interp_points = torch.cat((x0y0, x0y1, x1y0, x1y1), dim=2)
    # assert False, f"interp_points.device {interp_points.device} mask.device {mask.device}"    
    return interp_points,mask
    
    

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
    # points_2d_rounded = torch.round(points_2d)

    # Swap coordinates to match (row, column) format and return
    return torch.stack([points_2d[:, 1], points_2d[:, 0]], dim=-1)

def compute_2d_displacements(points_3d,velocity, intrinsic_matrices,pose_matrices, time_step=1):
    # Compute 3D positions in the next frame
    next_points_3d = points_3d+velocity*time_step
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
