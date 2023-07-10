import torch
import torch.nn as nn
import torchvision.models as models
from architecture.building_blocks import Swish, ResidualBlock

class SlowOnly(nn.Module):
    def __init__(self, slowfast):
        super().__init__()
        self.slowfast = slowfast

    def forward(self, x):
        # Use only the slow pathway (first half of the input)
        x_slow = x
        x_slow = self.slowfast.s1(x_slow)
        x_slow = self.slowfast.s2(x_slow)
        x_slow = self.slowfast.s3(x_slow)
        x_slow = self.slowfast.s4(x_slow)
        x_slow = self.slowfast.s5(x_slow)
        x_slow = self.slowfast.head(x_slow)
        return x_slow


class SlowfastVideoEncoder(nn.Module):
    def __init__(self,in_dim,out_dim, device="cuda", **kwargs):
        super().__init__()
        self.slowfast = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True).to(device=device)
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
        self.deeplab = torch.hub.load('pytorch/vision:v0.10.0','deeplabv3_resnet101', pretrained=True).to(device=device)
        self.deeplab.classifier[-1] = nn.Identity()

    def forward(self, x):
        return self.deeplab(x)['out']

class VectorEncoder(nn.Module):
    def __init__(self,in_dim,inner_dims,out_dim,normalize_output=True):
        super().__init__()
        layers = []
        for i in range(len(inner_dims)):
            if i == 0:
                layers.append(nn.Linear(in_dim,inner_dims[i]))
                layers.append(nn.InstanceNorm1d(inner_dims[i],affine=False,track_running_stats=False),
)
            else:
                layers.append(nn.Linear(inner_dims[i-1],inner_dims[i]))
            layers.append(Swish())
            layers.append(ResidualBlock(inner_dims[i]))
        layers.append(nn.Linear(inner_dims[-1],out_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
        
class VideoVectorEncoder(VectorEncoder):
    def __init__(self,in_dim,inner_dims,out_dim,decimation_size=32):
        layers = []
        # assert False, f"inner_dims {inner_dims}"
        in_data = torch.randn((1,in_dim))
        in_data.requires_grad=False
        # assert False, f"in_data.shape {in_data.shape}"
        for i in range(5):
            # if i == 0:
            layers.append(nn.Conv1d(1,1,5,stride=3,dilation=1))
            # else:
                # layers.append(nn.Linear(inner_dims[i-1],inner_dims[i]))
            layers.append(Swish())
            # layers.append(ResidualBlock(inner_dims[i]))
        # layers.append(nn.Linear(inner_dims[-1],out_dim))
        downsize_net = nn.Sequential(*layers)

        out_shape = downsize_net(in_data).shape
        super().__init__(out_shape[1],inner_dims,out_dim,normalize_output=False)
        self.downsize_net = downsize_net
    def forward(self, x):
        # decimated_in_dim = x.shape[0]//32
        # assert False, f"x.shape {x.shape}"
        x = x.unsqueeze(1)
        x = self.downsize_net(x)
        x = self.network(x).squeeze()
        # assert False, f"x.shape {x.shape}"
        return x
