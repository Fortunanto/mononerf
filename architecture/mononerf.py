from architecture.encoders import *
from architecture.residual_mlp import *
from architecture.generalizeable_dynamic_field import *
from dataloader import *
import torch
import torch.nn as nn
import time
# Path: train.py

class MonoNeRF(nn.Module):
    def __init__(self,slowfast_encoded_video) -> None:
        super().__init__()
        self.video_base_encoding = slowfast_encoded_video
        in_feature_dim = slowfast_encoded_video.shape[0]
        feature_dim = 2048
        self.video_encoder = ImplicitNet(in_feature_dim, [2048, 512, 256, 256], d_out=feature_dim).to(device="cuda")
        # self.implicit_net = ImplicitNet(2048, [512, 256, 256], d_out=256).to(device="cuda")
        self.ray_bending_estimator = PointTrajectory(3,feature_dim)

        self.images = torch.Tensor(load_images_as_tensor("data/ball/image"))
        
    def forward(self, index):
        
        downsampled_encoded_video = self.video_encoder(self.video_base_encoding)
        self.ray_bending_estimator(x,downsampled_encoded_video,)

            def forward(self, initial_point, features,intrinsics,poses,time_span=100):

        # assert False, f"downsampled_encoded_video.shape: {downsampled_encoded_video.shape}"
        pass

def train(model, train_loader, optimizer, criterion, device):
    pass


def wait_and_print(seconds):
    for i in range(1, seconds+1):
        print(f"{i} second passed")
        time.sleep(1)


def main():
    encoded_video = torch.load("data/ball/data/video_embedding.pt").to(device="cuda")
    model = MonoNeRF(encoded_video).to(device="cuda")
    # wait_time = 10
    # wait_and_print(wait_time)
    start_time = time.time()
    model(None)
    end_time = time.time()    
    execution_time = end_time - start_time
    print(f"The function took {execution_time} seconds to execute.")
    assert False, f"encoded_video.shape: {encoded_video.shape} execution_time {execution_time}"

if __name__ == '__main__':
    main()
