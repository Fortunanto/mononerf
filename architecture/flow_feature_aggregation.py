import torch
import torch.nn as nn
from architecture.building_blocks import SinActivation, Swish, ResidualBlock
import torch.nn.functional as F
from util.utils_3d import get_interpolation_indices, interpolate_feature_map
from einops import rearrange

class SpatialFeatureAggregation(nn.Module):
    def __init__(self,out_dim,image_features):
        super(SpatialFeatureAggregation, self).__init__()
        self.image_features = image_features
        spatial_feature_dim = image_features.shape[1]
        self.pre = nn.Linear(spatial_feature_dim, out_dim)
        self.cur = nn.Linear(spatial_feature_dim, out_dim)
        self.post = nn.Linear(spatial_feature_dim, out_dim)
        self.concatenated_features_comb = nn.Linear(3*out_dim,out_dim)
    def forward(self, trajectory,h,w,indices):
        features_interp = interpolate_feature_map(trajectory,h,w,indices,self.image_features)
        features_interp = rearrange(features_interp,'(time batch_size) feature_dim -> time batch_size feature_dim',batch_size=trajectory.shape[1])
        pre = self.pre(features_interp[0])
        cur = self.cur(features_interp[1])
        post = self.post(features_interp[2])
        concatenated_features = torch.cat([pre,cur,post],dim=-1)
        return self.concatenated_features_comb(concatenated_features)
        # assert False, f"features_interp shape: {features_interp.shape} device {features_interp.device}"

        # assert False, f"features_interp shape: {features_interp.shape} device {features_interp.device}"
        
        # assert False, f"interp_features shape: {interp_features.shape}"
        # assert False, f"features_topleft shape: {features_topleft.shape} features_topright shape: {features_topright.shape} features_bottomleft shape: {features_bottomleft.shape} features_bottomright shape: {features_bottomright.shape}"
        # assert False,f"indices shape: {indices.shape}"
        features = self.image_features[indices]

        assert False, f"interp shape: {interp.shape}"
        # assert False, f"trajectory_upper shape: {trajectory_upper.shape} trajectory_lower shape: {trajectory_lower.shape}"
        # indices = indices.cpu()
        # self.image_features = self.image_features[indices]
        
        output = F.grid_sample(self.image_features, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # assert False, f"trajectory_normalized shape: {trajectory_normalized.shape}"

        assert False, f"grid shape: {grid.shape}"

        # now output will be a tensor of shape [batch_size, dim, h, w], but all spatial positions will have
        # the same value, which is the value of the single point you wanted to sample.
        # you can take any spatial position and it should have the sampled value.
        sampled_value = output[..., 0, 0]  # take top-left corner as representative
        assert False, f"output shape {output.shape}, sampled_value shape {sampled_value.shape}"
        pre = self.pre(trajectory[0])
        cur = self.cur(trajectory[1])
        post = self.post(trajectory[2])
    
        return pre + cur + post