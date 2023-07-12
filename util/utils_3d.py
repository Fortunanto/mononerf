import torch
from einops import *

import numpy as np

def adjust_to_image_coordinates(points_2d, cx,cy):
    # points_2d[..., 0] = cx - points_2d[...,0]  # Add half the width to the x-coordinate
    # points_2d[..., 1] = cy - points_2d[...,1]   # Subtract y-coordinate from half the height

    return points_2d

def interpolate_feature_map(points,h,w,indices,features):
    scene_indices, image_indices = indices
    # assert False, f"scene_indices shape: {scene_indices.shape} image_indices shape: {image_indices.shape} features shape: {features.shape}"
    interp,mask = get_interpolation_indices(points=points, h=h, w=w)
    interp_shape,mask_shape = interp.shape,mask.shape
    if len(interp_shape)==5:
        mask = rearrange(mask,'b time n_samples -> (b time n_samples)')
        interp = rearrange(interp.to(features.device),'b time n_samples bounds xy -> (b time n_samples) bounds xy')
    else:
        mask = rearrange(mask,'n_samples b -> (n_samples b)')
        interp = rearrange(interp.to(features.device),'b n_samples bounds xy -> (b n_samples) bounds xy')
    # assert False, f"interp shape: {interp.shape}"
    image_indices = image_indices.unsqueeze(2).expand(-1,-1,points.shape[2]).reshape(-1).to(features.device)
    scene_indices = scene_indices.unsqueeze(2).expand(-1,-1,points.shape[2]).reshape(-1).to(features.device)
    out_shape = points.shape[:3]
    if len(interp_shape)==5:
        points = rearrange(points,'b time n_samples xy -> (b time n_samples) xy')
    else: 
        points = rearrange(points,'b n_samples xy -> (b n_samples) xy')
    # assert False, f"indices shape: {indices.shape} interp shape: {interp.shape}"
    y0x0 = interp[:,0]
    y0x1 = interp[:,1]
    y1x0 = interp[:,2]
    y1x1 = interp[:,3]
    image_indices = torch.clamp(image_indices,0,11)
    # try:
    features_topleft = features[scene_indices,image_indices,:,y0x0[:,0],y0x0[:,1]].to(points.device)
    features_topright = features[scene_indices,image_indices,:,y0x1[:,0],y0x1[:,1]].to(points.device)
    features_bottomleft = features[scene_indices,image_indices,:,y1x0[:,0],y1x0[:,1]].to(points.device)
    features_bottomright = features[scene_indices, image_indices,:,y1x1[:,0],y1x1[:,1]].to(points.device)
    
    y0x0 = y0x0.to(points.device)
    y0x1 = y0x1.to(points.device)
    y1x0 = y1x0.to(points.device)
    y1x1 = y1x1.to(points.device)

    wa = (((points[:,1]-y0x0[:,0])*(points[:,0]-y0x0[:,1]))).unsqueeze(1)
    wb = ((points[:,1]-y0x1[:,0])*(y0x1[:,1]-points[:,0])).unsqueeze(1)
    wc = ((y1x0[:,0]-points[:,1])*(points[:,0]-y1x0[:,1])).unsqueeze(1)
    wd = ((y1x1[:,0]-points[:,1])*(y1x1[:,1]-points[:,0])).unsqueeze(1)

    interp_features = wa * features_topleft + wb * features_topright + wc * features_bottomleft + wd * features_bottomright
    if torch.isnan(interp_features).any():
        assert False, f"interp_features: {interp_features}"
    return (mask.unsqueeze(1)*interp_features).reshape(*out_shape,-1)
# except:
    #     assert False, f"y0x0[:,0] max {y0x0[:,0].max()} min {y0x0[:,0].min()} y0x0[:,1] max {y0x0[:,1].max()} min {y0x0[:,1].min()} y0x1 \
    #     [:,0] max {y0x1[:,0].max()} min {y0x1[:,0].min()} y0x1[:,1] max {y0x1[:,1].max()} min {y0x1[:,1].min()} y1x0[:,0] max {y1x0[:,0].max()} min \
    #     {y1x0[:,0].min()} y1x0[:,1] max {y1x0[:,1].max()} min {y1x0[:,1].min()} y1x1[:,0] max {y1x1[:,0].max()} min {y1x1[:,0].min()} y1x1[:,1] max \
    #     {y1x1[:,1].max()} min {y1x1[:,1].min()}"
def get_interpolation_indices(points,h,w):
    y0 = torch.floor(points[...,1]).long()
    y1 = y0 + 1
    x0 = torch.floor(points[...,0]).long()

    x1 = x0 + 1
    mask = (((y0 < 0) | (y1 >= h) | (x0 < 0) | (x1 >= w))).int()
    x0 = torch.clamp(x0,0,w-1)
    x1 = torch.clamp(x1,0,w-1)
    y0 = torch.clamp(y0,0,h-1)
    y1 = torch.clamp(y1,0,h-1)  
    y0x0 = torch.stack((y0,x0), dim=-1).unsqueeze(3)
    y0x1 = torch.stack((y0, x1), dim=-1).unsqueeze(3)
    y1x0 = torch.stack((y1, x0), dim=-1).unsqueeze(3)
    y1x1 = torch.stack((y1, x1), dim=-1).unsqueeze(3)

    
    interp_points = torch.cat([y0x0,y0x1,y1x0,y1x1], dim=3)
    # assert False, f"interp_points.device {interp_points.device} mask.device {mask.device}"    
    return interp_points,1-mask
  
def project_3d_to_2d(points_3d,pose_matrix, intrinsic_matrix):
    """
    Project 3D points from world coordinates to 2D image coordinates.

    :param points_3d: 3D points in world coordinates as a tensor of shape [B, 3].
    :param intrinsic_matrix: Camera intrinsic matrix as a tensor of shape [B, 4, 4].
    :param pose_matrix: Camera pose (extrinsic) matrix as a tensor of shape [B, 4, 4].
    :return: 2D points in image coordinates as a tensor of shape [B, 2].
    """
    if len(points_3d.shape) == 4:
        points_3d = rearrange(points_3d, 'b t n_samples xyz -> (b t n_samples) xyz')
        pose_matrix = rearrange(pose_matrix, 'b t n_samples x y -> (b t n_samples) x y')
        intrinsic_matrix = rearrange(intrinsic_matrix, 'b t n_samples x y -> (b t n_samples) x y')
    else:
        points_3d = rearrange(points_3d, 'b n_samples xyz -> (b n_samples) xyz')
        pose_matrix = rearrange(pose_matrix, 'b n_samples x y -> (b n_samples) x y')
        intrinsic_matrix = rearrange(intrinsic_matrix, 'b n_samples x y -> (b n_samples) x y')

    assert points_3d.dim() == 2
    assert intrinsic_matrix.dim() == 3
    assert pose_matrix.dim() == 3
    pose_matrix = pose_matrix.inverse()
    # intrinsic_matrix = intrinsic_matrix.inverse()
    # Convert to homogeneous coordinates
    points_3d_hom = torch.cat([points_3d, torch.ones_like(points_3d[..., :1])], dim=-1)

    # Transform 3D points to camera coordinates
    points_cam_hom = torch.einsum('bik, bk -> bi', pose_matrix, points_3d_hom)

    # Project points onto 2D image plane
    # points_2d_hom = torch.einsum('bik, bk -> bi', intrinsic_matrix, points_cam_hom)

    # Convert back to Cartesian coordinates
    points_2d = points_cam_hom[..., :2] / points_cam_hom[..., 2:3]
    points_2d[...,0] = -points_2d[...,0]*intrinsic_matrix[:, 0, 0] + intrinsic_matrix[:, 0, 2]
    points_2d[...,1] = points_2d[...,1]*intrinsic_matrix[:, 1, 1] + intrinsic_matrix[:, 1, 2]
    # cx = intrinsic_matrix[:, 0, 2]
    # cy = intrinsic_matrix[:, 1, 2]
    # points_2d[...,1] = points_2d[...,1]*-1
    # points_2d[...,1] -= 2*cy
    return points_2d
    # return torch.cat((points_2d[...,1:2],points_2d[...,0:1]), dim=-1)

def project_3d_to_2d_batch(points_3d, intrinsic_matrix, pose_matrix):
    """
    Project a batch of 3D points from world coordinates to 2D image coordinates.

    :param points_3d: Batch of 3D points in world coordinates as a tensor of shape [B, 3].
    :param intrinsic_matrix: Batch of camera intrinsic matrices as a tensor of shape [B, 3, 3].
    :param pose_matrix: Batch of camera pose (extrinsic) matrices as a tensor of shape [B, 3, 4].
    :return: Batch of 2D points in image coordinates as a tensor of shape [B, 2], rounded to nearest integers.
    """
    # assert False, f"points_3d.shape: {points_3d.shape} intrinsic_matrix.shape: {intrinsic_matrix.shape} pose_matrix.shape: {pose_matrix.shape}"
    if len(points_3d.shape) == 4:
        points_3d = rearrange(points_3d, 't b n_samples xyz -> (t b n_samples) xyz')
        pose_matrix = rearrange(pose_matrix, 't b n_samples x y -> (t b n_samples) x y')
        intrinsic_matrix = rearrange(intrinsic_matrix, 't b n_samples x y -> (t b n_samples) x y')
    else:
        points_3d = rearrange(points_3d, 'b n_samples xyz -> (b n_samples) xyz')
        pose_matrix = rearrange(pose_matrix, 'b n_samples x y -> (b n_samples) x y')
        intrinsic_matrix = rearrange(intrinsic_matrix, 'b n_samples x y -> (b n_samples) x y')

    # Extend points_3d to homogeneous coordinates
    points_3d_homogeneous = torch.cat([points_3d, torch.ones(points_3d.size()[:-1]+(1,), device=points_3d.device)], dim=-1)
    # Apply extrinsic matrix
    pose_matrix = torch.inverse(pose_matrix)
    points_cam = torch.bmm(pose_matrix, points_3d_homogeneous.unsqueeze(-1)).squeeze(-1)
    # points_cam = 
    # Apply intrinsic matrix
    intrinsic_matrix = torch.inverse(intrinsic_matrix)
    # assert False, f"intrinsic_matrix[0]\n\n  {intrinsic_matrix[0]} pose_matrix[0]\n\n {pose_matrix[0]}"
    points_2d_homogeneous = torch.bmm(intrinsic_matrix, points_cam.unsqueeze(-1)).squeeze(-1)
    # assert False, f"points_cam[0]: {points_cam[0]} points_2d_homogeneous[0] {points_2d_homogeneous[0]}"

    # Normalize to get actual 2D coordinates
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2].unsqueeze(1)
    cx = intrinsic_matrix[:, 0, 2]
    cy = intrinsic_matrix[:, 1, 2]
    points_2d = adjust_to_image_coordinates(points_2d, cx, cy)
    # assert False, f"intrinsic shape {intrinsic_matrix.shape} pose shape {pose_matrix.shape}"
    # points_2d =
    # Round the coordinates to nearest integers
    # points_2d_rounded = torch.round(points_2d)

    # Swap coordinates to match (row, column) format and return
    return torch.stack([points_2d[:, 1], points_2d[:, 0]], dim=-1)

# def compute_2d_displacements(points_start,points_end, intrinsic_matrices,pose_matrices, time_step=1):
#     # Project 3D points to 2D
#     points_2d = project_3d_to_2d(points_start,pose_matrices, intrinsic_matrices)
#     next_points_2d = project_3d_to_2d(points_end, pose_matrices,intrinsic_matrices)
#     # assert not points_3d.isnan().any(), f"Points_3d has NaN values: {points_3d}"
#     # assert not points_2d.isnan().any(), f"Points_2d has NaN values: {points_2d}"
#     # assert not next_points_2d.isnan().any(), f"Next_points_2d has NaN values: {next_points_2d}"
#     # Compute 2D displacements
#     return displacements_2d

if __name__ == "__main__":
    point = torch.randint(200,400,size=(10,2)).to(device="cuda")
    pose = torch.load("data/ball/data/pose.pt").to(device="cuda")
    intrinsic = torch.load("data/ball/data/intrinsics.pt").to(device="cuda")
