import torch
import torch.nn as nn

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            Swish(),
            nn.Linear(dim, dim),

        )

    def forward(self, input_tensor):
        shape = input_tensor.shape
        input_tensor = input_tensor.reshape(-1,shape[-1])
        out = input_tensor + self.block(input_tensor)
        return out.reshape(*shape)
