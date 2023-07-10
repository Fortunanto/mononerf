import torch
import torch.nn as nn
from architecture.building_blocks import SinActivation, Swish, ResidualBlock
import torch.nn.functional as F
from util.utils_3d import get_interpolation_indices, interpolate_feature_map
from einops import rearrange

class SpatialFeatureAggregation(nn.Module):
    def __init__(self,in_dim,out_dim,image_features):
        super(SpatialFeatureAggregation, self).__init__()
        self.image_features = image_features
        self.pre = nn.Linear(in_dim,out_dim)
        self.cur = nn.Linear(in_dim,out_dim)
        self.post =nn.Linear(in_dim,out_dim)
        self.concatenated_features_comb = nn.Linear(3*out_dim,out_dim)
    
    def forward(self, trajectory,h,w,indices):
        features_interp = interpolate_feature_map(trajectory,h,w,indices,self.image_features)
        pre = self.pre(features_interp[:,0])
        cur = self.cur(features_interp[:,1])
        post = self.post(features_interp[:,2])
        concatenated_features = self.concatenated_features_comb(torch.cat([pre,cur,post],dim=-1))
        if torch.isnan(concatenated_features).any():
            assert False, f"concatenated_features: {concatenated_features}"
        return concatenated_features,pre,cur,post
