import torch
import torch.nn as nn
import torchvision.models as models
from .residual_mlp  import *

class SlowfastVideoEncoder(nn.Module):
    def __init__(self,in_dim,out_dim, device="cuda", **kwargs):
        super().__init__()
        self.slowfast = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True).to(device=device)
        self.slowfast.blocks[-1] = IdentitySqueezeReshape()
        self.linear = nn.Linear(112896*4, out_dim).to(device=device)

    def forward(self, x):
        return self.linear(self.slowfast(x))
        # assert False, f"output.shape: {output.shape}"

class IdentitySqueezeReshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze().reshape(-1)


def get_video_encoder(device="cuda",**kwargs):
    # Choose the `slowfast_r50` model 
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True).to(device=device)

    # Modify the last layer to an identity layer
    model.blocks[-1] = IdentitySqueezeReshape()

    # Freeze all the model parameters except the modified last layer
    for param in model.parameters():
        param.requires_grad = False

    return model

if __name__ == "__main__":
    model = SlowfastVideoEncoder(in_dim=10,out_dim=2048)
    print(model)
    input_tensor_slow = torch.randn(4, 3, 8, 448, 448).to(device="cuda")  # Example input tensor for slow pathway
    input_tensor_fast = torch.randn(4, 3, 32, 448, 448).to(device="cuda")  # Example input tensor for fast pathway
    output_tensor = model([input_tensor_slow, input_tensor_fast])
    print(output_tensor.shape)
