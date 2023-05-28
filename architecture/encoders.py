import torch
import torch.nn as nn
import torchvision.models as models

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
    
class ResnetImageEncoder(nn.Module):
    def __init__(self,in_dim,out_dim, device="cuda", **kwargs):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True).to(device=device)
        
        # self.resnet.avgpool = nn.Identity()
        # self.resnet.fc = nn.Identity()
        # assert False, f"resnet keys: {self.resnet}"
        for param in self.resnet.parameters():
            param.requires_grad = False
        # self.resnet.fc = nn.Linear(512, out_dim).to(device=device)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        assert False, f"x.shape: {x.shape}"
        x = self.avgpool(x)
    
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x    
