import torch
import torch.nn as nn
from architecture.mononerf import NeRF
class ImplicitStaticField(nn.Module):
    def __init__(self,point_dim=60 ,feature_dim=256):
        super(ImplicitStaticField, self).__init__()
        self.feature_dim = feature_dim
        self.point_dim = point_dim
        self.nerf = NeRF(input_ch = 256+60)
        self.feature_layer = nn.Linear(feature_dim,feature_dim)
    
    def forward(self,point,features):
        