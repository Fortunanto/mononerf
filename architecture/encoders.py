import torch
import torch.nn as nn
import torchvision.models as models
from architecture.building_blocks import Swish, ResidualBlock

class SlowfastVideoEncoder(nn.Module):
    def __init__(self,in_dim,out_dim, device="cuda", **kwargs):
        super().__init__()
        self.slowfast = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True).to(device=device)
        self.slowfast.blocks[-1] = IdentitySqueezeReshape()

    def forward(self, x):
        return self.slowfast(x)
        # assert False, f"output.shape: {output.shape}"

class IdentitySqueezeReshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze().reshape(-1)

class DeeplabV3Encoder(nn.Module):
    def __init__(self,in_dim,out_dim, device="cuda", **kwargs):
        super().__init__()
        self.deeplab = torch.hub.load('pytorch/vision:v0.10.0','deeplabv3_resnet50', pretrained=True).to(device=device)
        self.deeplab.classifier[-1] = nn.Identity()

    def forward(self, x):
        return self.deeplab(x)['out']

class VectorEncoder(nn.Module):
    def __init__(self,in_dim,inner_dims,out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, 2048),
            Swish(),
            ResidualBlock(2048),
            nn.Linear(2048, 1024),
            Swish(),
            ResidualBlock(1024),
            nn.Linear(1024, out_dim),
        )
    def forward(self, x):
        return self.network(x)
        
