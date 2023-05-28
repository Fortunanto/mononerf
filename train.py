import torch
import torch.nn as nn
import os
from util.config import *
from architecture.generalizeable_dynamic_field import *
from architecture.encoders import *
from dataloader import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.ray_helpers import *
from einops import *
from loss import *
import wandb
import copy
import torch
import torch.autograd as autograd
from torchviz import make_dot

# PYZSHCOMPLETE_OK
def check_parameters_changed(model, prev_parameters):
    current_parameters = copy.deepcopy(model.state_dict())
    if prev_parameters is None:
        return True
    else:
        for param_name in prev_parameters.keys():
            if not torch.equal(prev_parameters[param_name], current_parameters[param_name]):
                return True
        return False

def train(ray_bender,video_downsampler,nerf, train_loader, optimizer, criterion,config, device):
    # flow_loss = 0
    epoch_flow_loss = 0
    num_batches = 0

    prev_parameters = None
    # ray_bender.train()
    criterion = torch.nn.MSELoss()
    for epoch in range(config.epochs):
        ray_bender.train()

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}") as pbar:
            for batch_idx, (index, target) in enumerate(tqdm(train_loader)):
                # optimi/zer.zero_grad()
                target = target.to(device)
                image_indices,x,y = index
                image_indices = image_indices.unsqueeze(1).to(device="cuda")  # reshape from (n,) to (n, 1)
                image_indices = torch.cat((image_indices-1,image_indices,image_indices+1),dim=1)
                image_indices_reshaped = image_indices.reshape(-1)
                pose,intrinsics = train_loader.dataset.get_pose_intrinsics(image_indices_reshaped)
                flows = train_loader.dataset.get_video_flows()
                flows = F.interpolate(flows,size=(240,360))
                pose,intrinsics = pose.reshape(*image_indices.shape,4,4) , intrinsics.reshape(*image_indices.shape,4,4)
                rays_o,rays_d = get_rays_from_points((x,y),intrinsics[:,1],pose[:,1])

                points = get_points_along_rays(rays_o,rays_d,config.near,config.far,config.linear_displacement,config.perturb,config.n_samples)
                trajectory = ray_bender(points,video_downsampler(train_loader.dataset.get_video_embedding()),intrinsics,pose,image_indices[:,1],time_span=config.time_span)
                # assert False, f"trajectory.shape: {trajectory.shape}"
                flow_loss = supervise_flows(trajectory,flows[image_indices[:,1],:,x,y],pose[:,1],intrinsics[:,1],criterion)
                # assert False, f"ray_bender.parameters(): {list(ray_bender.parameters())}"
                flow_loss.backward(retain_graph=True)
                dot = make_dot(flow_loss, params=dict(loss=flow_loss))
                dot.render(filename='gradient_graph_2', format='png')

                optimizer.step()
                # # wandb.log({"flow_loss": flow_loss.item()})

                epoch_flow_loss += flow_loss.item()
                num_batches += 1
                pbar.set_postfix({"Flow Loss": flow_loss.item()})
                pbar.update()


                # assert False, f"initial_point_indices_2d: {initial_point_indices_2d}"
                # def forward(self, initial_point, features,intrinsics,poses,time_span=100):

                # assert False, f"image_indices {image_indices} x {x} y {y}"
                # optimizer.zero_grad()
                # output = model(data)
                # loss = criterion(output, target)
                # loss.backward()
                # optimizer.step()
            # Update previous parameters
            epoch_flow_loss /= num_batches
            # wandb.log({"epoch_flow_loss": epoch_flow_loss})
            epoch_flow_loss = 0
            num_batches = 0

        

def main():
    config = get_config()
    # encoded_data_path = os.path.join(config.data_folder,"data")
    dataset = CustomEncodingsImageDataset(config.data_folder)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # video_embedding_path = os.path.join(encoded_data_path,"video_embedding.pt")
    video_embedding = dataloader.dataset.get_video_embedding()
    video_downsampler = ImplicitNet(video_embedding.shape[0], [2048,2048, 2048, 2048], d_out=2048).to(device="cuda")
    ray_bending_estimator = PointTrajectory(3,2048).to(device="cuda")
    for param in ray_bending_estimator.parameters():
        print(f"param {param}")
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)

    # video_downsampler.train()
    # wandb.init(project=config.exp_group_name, config=config)
    # wandb.watch(ray_bending_estimator)
    # wandb.watch(video_downsampler)
    optimizer = torch.optim.SGD(ray_bending_estimator.parameters(), lr=config.lr)
    for param in ray_bending_estimator.parameters():
        param.requires_grad=True
    # assert False, f"ray_bending_estimator.parameters() {ray_bending_estimator.parameters()}"
    train(ray_bending_estimator,video_downsampler,None,dataloader,optimizer,None,config,"cuda")
    pass
    
if __name__ == '__main__':
    main()
