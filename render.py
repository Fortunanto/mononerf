import torch
import torch.nn as nn
import os
from util.config import *
from architecture.generalizeable_dynamic_field import *
from architecture.encoders import *
from architecture.flow_feature_aggregation import *
from dataloader import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.ray_helpers import *
from einops import *
from loss import *
import torch
from architecture.mononerf import *
from util.render_utils import batchify_rays

def create_dataset_and_networks(config):
    dataset = CustomEncodingsImageDataset(config.data_folders,config,load_images=True)
    networks = {}
    video_embedding = dataset.get_video_embedding()
    networks["video_downsampler"] = VectorEncoder(
        video_embedding.shape[1], [256, 512], 256,normalize_output=False).to(device="cuda")
    networks["spatial_encoder"] = VectorEncoder(256, [256, 512], 256).to(device="cuda")
    networks["ray_bender"] = PointTrajectoryNoODE(3, 256).to(device="cuda")
    networks["spatial_feature_aggregation"] = SpatialFeatureAggregation(dataset.image_encodings.shape[2],
                                                                      256, dataset.image_encodings).to(device="cuda")
    networks["nerf"] = NeRF(D=config.architecture.layer_count_dynamic,
                          skips=config.architecture.dynamic_layer_skips,input_ch=256*2+60+10)\
                        .to(device="cuda")
    networks["static_field_NeRF"] = NeRF(D=config.architecture.layer_count_static,skips=config.architecture.static_layer_skips,
        input_ch=256+60, dynamic=False).to(device="cuda")
    # ray_bender.load_state_dict(torch.load("/home/yiftach/main/Research/MonoNeRF/checkpoints/fancy-dust-735/ray_bender.pt"))
    networks["static_encoder"] = VectorEncoder(512,[256],256,normalize_output=True).to(device="cuda")
    return dataset,networks
def load_checkpoints(model_dict,checkpoint_dir):
    for model_name in model_dict:
        model_dict[model_name].load_state_dict(torch.load(os.path.join(checkpoint_dir,f"{model_name}.pt")))
    return model_dict
    
def main():
    config = get_config()
    config.set_run_name("classic-donkey-812")
    config.set_run_name("classic-donkey-812")
    return
    dataset,networks_dict = create_dataset_and_networks(config)
    networks_dict = load_checkpoints(networks_dict,os.path.join(config.checkpoint_dir,"classic-donkey-812"))
    for t in tqdm(range(1,10)):
        c2w = dataset.poses[0,3].cpu().numpy()
        K = dataset.intrinsics[0,0].cpu().numpy()
        rays_o,rays_d = get_rays_np(540/2,960/2,K,c2w)
        rays_o, rays_d = ndc_rays_np(540/2,960/2,K[0,0], 1., rays_o, rays_d)
        image = batchify_rays(rays_o,rays_d,networks_dict,dataset,output_dir=os.path.join("results","classic-donkey-812"),t=t,verbose=False)
        print(image)
    # grad_params += list(spatial_encoder.parameters())
    # grad_params += list(ray_bender.parameters())
    # grad_params += list(spa.parameters())
    # grad_params += list(nerf.parameters())
    # grad_params += list(static_field_NeRF.parameters())
    # grad_params += list(static_encoder.parameters())

    # total_params = sum(p.numel() for p in grad_params)
    # formatted_params = format_number(total_params)

    # print("---------------------------------------------------------------")
    # print("---------------------------------------------------------------")
    # print("                      NETWORK PARAMETER COUNT                  ")
    # print("---------------------------------------------------------------")
    # print("      Total number of parameters: {:^20}     ".format(
    #     formatted_params))
    # print("---------------------------------------------------------------")
    # print("---------------------------------------------------------------")

    # part_percentages = []
    # for part in ['video_downsampler', 'spatial_encoder', 'ray_bender', 'spa', 'nerf', 'static_field_NeRF', 'static_encoder']:
    #     part_params = sum(p.numel() for p in eval(part).parameters())
    #     part_percentage = (part_params / total_params) * 100
    #     part_percentages.append((part, part_percentage))

    # part_percentages.sort(key=lambda x: x[1], reverse=True)

    # print("Percentage of parameters per part of the network:")
    # for part, percentage in part_percentages:
    #     print("{:<25} {:>10.2f}%".format(part, percentage))

    # print("---------------------------------------------------------------")
    # print("---------------------------------------------------------------")
    # for param in grad_params:
    #     if param.dim() > 1:
    #         nn.init.xavier_uniform_(param)
    #     else:
    #         nn.init.zeros_(param)

    # optimizer = torch.optim.Adam(grad_params, lr=config.lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = config.lr_milestones,gamma = config.lr_gamma)
    # train(ray_bender, video_downsampler, spa, spatial_encoder, nerf, static_field_NeRF,
    #         static_encoder, dataloader, scheduler,optimizer, torch.nn.MSELoss(), config, "cuda")


def format_number(number):
    units = ["", "K", "M", "B"]
    unit_index = 0
    while number >= 1000 and unit_index < len(units) - 1:
        number /= 1000
        unit_index += 1
    formatted_number = "{:,.2f}{}".format(number, units[unit_index])
    return formatted_number


if __name__ == '__main__':
    main()
