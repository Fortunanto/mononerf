import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 
from tqdm import tqdm
import torch.nn.functional as F
import cv2 as cv
from util.ray_helpers import get_rays_np,ndc_rays_np,get_rays,ndc_rays



def load_images_as_tensor(directory):
    images = []
    
    # Iterate over each file in the directory
    files = sorted(os.listdir(directory))
    # assert False, f"os.listdir(directory) {files}"
    for filename in files:
        
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Open the image using PIL
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            # Convert the image to numpy array
            image_array = np.array(image)
            
            # Add the image array to the list
            images.append(image_array)
    
    # Convert the list of image arrays to a numpy tensor
    images_tensor = np.stack(images)
    if len(images_tensor.shape)==4:
        images_tensor = images_tensor.transpose(3,0,1,2)
    return images_tensor

def load_disparities_as_tensor(directory):
    disparities = []
    
    files = sorted(os.listdir(directory))  
    for filename in files:
        disp_path = os.path.join(directory, filename)
        disparity = np.load(disp_path)
        disparities.append(disparity)
    disparities = np.stack(disparities)
    return disparities

def load_and_process_flows(directories, flow_direction,interpolation,downscale):
    flows = [load_flows_as_tensor(os.path.join(dir, "flow"), flow_direction) for dir in directories]
    interpolation_func = lambda x: interpolation(x,downscale)
    return torch.from_numpy(np.array(extract_flows(flows,interpolation_func))).float(), torch.from_numpy(np.array(extract_masks(flows,interpolation_func))).float()

def load_flows_as_tensor(directory, filter):
    flows = []
    files = sorted([f for f in os.listdir(directory) if filter in f])
    for filename in files:
        flow_path = os.path.join(directory, filename)
        flow_mask = np.load(flow_path)
        flow = flow_mask["flow"].transpose(2,0,1)
        mask = flow_mask["mask"]
        flows.append((flow, mask))
    return flows

def extract_flows(flows_list,interpolation_func):
    return [[interpolation_func(flow[0][np.newaxis,...]) for flow in flows] for flows in flows_list]

def extract_masks(flows_list,interpolation_func):
    return [[interpolation_func(flow[1][np.newaxis,...]) for flow in flows] for flows in flows_list]

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def normalize(x):
    return x / (np.linalg.norm(x))

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def interpolate(images,downscale_factor):
    if isinstance(images, np.ndarray):
        is_numpy = True
        images = torch.from_numpy(images).float()
    else:
        is_numpy = False
    if len(images.shape)<4:
        images = images.unsqueeze(1)
    # interpolation = images[]
    # interpolation = images[...,140:500,220:560]
    interpolation = F.interpolate(images, scale_factor=1/downscale_factor, mode="bilinear", align_corners=False).squeeze()
    if is_numpy:
        interpolation = interpolation.numpy()
    
    return interpolation

def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def calculate_sc(pose_bound,bd_factor):
    return 1. if bd_factor is None else 1./(np.percentile(pose_bound[:, -2], 5) * bd_factor)


import itertools
class CustomEncodingsImageDataset(torch.utils.data.Dataset):
    def __init__(self, dirs,config,bd_factor=.9, transform=None,load_images=True):
        if config.dataset_type == "dynamic scene dataset":
           self.__load_dynamic_scene_dataset(dirs,config,bd_factor,transform,load_images)
        else:
           self.__load_pacnerf_dataset(dirs,config,bd_factor,transform,load_images)

        self.indices = list(itertools.product(range(0,self.n_scenes), range(0,self.n_images),range(1,self.h-1),range(1,self.w-1)))        
        # self.indices = list(itertools.product(range(0,self.n_scenes), range(3,5),range(self.h//2,self.h//2+90),range(self.w//2,self.w//2+90)))        
        scene_ray_o = []
        scene_ray_d = []
        for scene in range(self.n_scenes):
            images_ray_o = []
            images_ray_d = []
            for image in range(self.n_images):
                intrinsic = self.intrinsics[scene][image].cpu().detach().numpy()
                pose = self.poses[scene][image].cpu().detach().numpy()
                rays_o,rays_d = get_rays_np(self.h, self.w, intrinsic, pose)
                if config.forward_facing:
                    rays_o, rays_d = ndc_rays_np(self.h, self.w,
                        intrinsic[0,0], 1., rays_o, rays_d)
                rays_o = rays_o.transpose(2,0,1)
                rays_d = rays_d.transpose(2,0,1)
                images_ray_o.append(rays_o)
                images_ray_d.append(rays_d)
            scene_ray_o.append(images_ray_o)
            scene_ray_d.append(images_ray_d)
        self.rays_o_np = np.array(scene_ray_o)
        self.rays_d_np = np.array(scene_ray_d)
        # object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        # object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])

        # self.object_bbox_min = object_bbox_min[:, None]
        # self.object_bbox_max = object_bbox_max[:, None] 
    def __load_pacnerf_dataset(self,dirs,config,bd_factor=.9, transform=None,load_images=True):
        self.data_dir = [os.path.join(dir,"2") for dir in dirs]
        self.scene_names = [os.path.basename(os.path.normpath(dir)) for dir in dirs]
        self.embedding_dirs = [os.path.join(dir,"embeddings") for dir in self.data_dir]
        self.image_dirs = [os.path.join(dir,"images") for dir in self.data_dir]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.disparities = torch.stack([interpolate(torch.Tensor(load_disparities_as_tensor(os.path.join(dir,"disp"))),config.downscale) for dir in self.data_dir])    
        self.flows_fwd = torch.stack([interpolate(torch.load(os.path.join(os.path.join(dir,"flows","forward","flows.pt"))),config.downscale) for dir in self.data_dir]).detach().to(device="cpu")
        self.flows_bwd = torch.stack([interpolate(torch.load(os.path.join(os.path.join(dir,"flows","backward","flows.pt"))),config.downscale) for dir in self.data_dir]).detach().to(device="cpu")
        self.flows_fwd.requires_grad = False
        self.flows_bwd.requires_grad = False
        self.images = torch.stack([interpolate(torch.Tensor(load_images_as_tensor(dir).transpose(1,0,2,3)),config.downscale)/255 for dir in self.image_dirs])
        self.flows_fwd_images = torch.stack([interpolate(torch.Tensor(load_images_as_tensor(os.path.join(dir,"flows","forward")).transpose(1,0,2,3)),config.downscale)/255 for dir in self.data_dir])
        self.flows_bwd_images = torch.stack([interpolate(torch.Tensor(load_images_as_tensor(os.path.join(dir,"flows","backward")).transpose(1,0,2,3)),config.downscale)/255 for dir in self.data_dir])

        self.images_masks = torch.stack([interpolate(torch.Tensor(load_images_as_tensor(os.path.join(dir,"masks"))),config.downscale) for dir in self.data_dir])
        self.images_masks = (self.images_masks  >128).float()
        self.image_encodings = [F.interpolate(torch.load(os.path.join(dir, "image_embeddings_dynamic.pt"))[...,70:250,110:280],mode="bilinear",scale_factor=2) for dir in self.embedding_dirs]
        self.image_encodings = torch.stack(self.image_encodings)
        mean = self.image_encodings.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0].min(dim=1,keepdim=True)[0]
        std = self.image_encodings.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
        mask = (std==0).expand_as(std)
        std[mask] = 1
        self.image_encodings = (self.image_encodings - mean) / std
        self.h,self.w=self.images.shape[-2],self.images.shape[-1]
        self.n_images = self.images.shape[1]
        self.n_scenes = self.images.shape[0]
        self.video_embeddings = [torch.load(os.path.join(dir, "video_embeddings.pt")).squeeze().to(device="cpu") for dir in self.embedding_dirs]
        # self.video_embeddings = [(enc - enc.mean()) for enc in self.video_embeddings]
        # self.video_embeddings = torch.stack([(enc /enc.max()) for enc in self.video_embeddings])
        self.video_embeddings = torch.stack(self.video_embeddings)
        self.video_embeddings = self.video_embeddings.detach()
        self.video_embeddings.requires_grad_(False)        
        poses = np.array([np.load(os.path.join(dir, "c2ws.npy")) for dir in self.data_dir])
        # sc = np.array([1. if bd_factor is None else calculate_sc(pose_bound,bd_factor) for pose_bound in poses_bounds]).reshape(-1,1,1)

        # poses[...,:3,3] *= sc
        # poses = np.concatenate([poses[..., 1:2],
                        #    -poses[..., 0:1],
                            # poses[...,2:]], -1)
        last_row = np.array([0,0,0,1])
        self.poses = torch.Tensor(np.concatenate([poses, np.tile(last_row, [poses.shape[0],poses.shape[1],1,1])], axis=2))
        intrinsics = np.array([np.load(os.path.join(dir, "intrinsics.npy")) for dir in self.data_dir])
        
        self.intrinsics = torch.Tensor(intrinsics)
        # self.intrinsics[...,0,2] = self.images[
        # self.intrinsics[...,0,0]= self.intrinsics[...,0,0]
        # self.intrinsics[...,1,1]= self.intrinsics[...,1,1]
        self.intrinsics[...,0,2]= self.images.shape[-1]/2
        self.intrinsics[...,1,2]= self.images.shape[-2]/2
        print("done loading dataset")
    def __load_dynamic_scene_dataset(self,dirs,config,bd_factor=.9, transform=None,load_images=True):
        self.data_dir = [os.path.join(dir,"data") for dir in dirs]
        self.scene_names = [os.path.basename(os.path.normpath(dir)) for dir in dirs]
        self.embedding_dirs = [os.path.join(dir,"embeddings") for dir in dirs]
        self.image_dirs = [os.path.join(dir,"images") for dir in dirs]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.disparities = torch.stack([interpolate(torch.Tensor(load_disparities_as_tensor(os.path.join(dir,"disp"))),config.downscale) for dir in dirs])    
 
        self.flows_fwd, self.flows_fwd_masks = load_and_process_flows(dirs, "fwd",interpolate,config.downscale)
        self.flows_bwd, self.flows_bwd_masks = load_and_process_flows(dirs, "bwd",interpolate,config.downscale)                
        if load_images:
            self.images = torch.stack([interpolate(torch.Tensor(load_images_as_tensor(dir).transpose(1,0,2,3)),config.downscale)/255 for dir in self.image_dirs])
            self.images_masks = (torch.stack([interpolate(torch.Tensor(load_images_as_tensor(os.path.join(dir,"motion_masks"))),config.downscale) for dir in dirs]) >128).float()
            self.image_encodings = [torch.load(os.path.join(dir, "image_embeddings_dynamic.pt")) for dir in self.embedding_dirs]
            self.image_encodings = torch.stack(self.image_encodings)
            mean = self.image_encodings.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0].min(dim=1,keepdim=True)[0]
            std = self.image_encodings.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
            mask = (std==0).expand_as(std)
            std[mask] = 1
            self.image_encodings = (self.image_encodings - mean) / std
            self.h,self.w=self.images.shape[-2],self.images.shape[-1]
            self.n_images = self.images.shape[1]
            self.n_scenes = self.images.shape[0]
        else:
            self.h,self.w = 540,960
            self.n_images=1
            self.n_scenes = 1
        self.video_embeddings = [torch.load(os.path.join(dir, "video_embeddings.pt")).squeeze().to(device="cpu") for dir in self.embedding_dirs]
        # self.video_embeddings = [(enc - enc.mean()) for enc in self.video_embeddings]
        # self.video_embeddings = torch.stack([(enc /enc.max()) for enc in self.video_embeddings])
        self.video_embeddings = torch.stack(self.video_embeddings)
        self.video_embeddings = self.video_embeddings.detach()
        self.video_embeddings.requires_grad_(False)
        # mean = self.video_embeddings.mean(dim=0,keepdim=True)
        # std = self.video_embeddings.std(dim=0,keepdim=True)

        # self.video_embeddings = (self.video_embeddings - mean) / std
        # mask = (std == 0).expand_as(self.video_embeddings)
        # self.video_embeddings[mask] = 0
        # self.image_encodings = 
        # self.disparities = [np.load(os.path.join(dir,, ".npy")) for dir in self.data_dir]

        
        poses_bounds = np.array([np.load(os.path.join(dir, "poses_bounds.npy")) for dir in dirs])
        poses = poses_bounds[:,:, :-2].reshape([poses_bounds.shape[0],-1, 3, 5])
        sc = np.array([1. if bd_factor is None else calculate_sc(pose_bound,bd_factor) for pose_bound in poses_bounds]).reshape(-1,1,1)

        poses[...,:3,3] *= sc
        hwfs = poses[:,:,:,-1]*1./config.downscale

        poses = np.concatenate([poses[..., 1:2],
                           -poses[..., 0:1],
                            poses[...,2:]], -1)
        poses = np.stack([recenter_poses(pose) for pose in poses])[...,:-1]
        # poses = np.stack([pose for pose in poses])[...,:-1]

        last_row = np.array([0,0,0,1])
        self.poses = torch.Tensor(np.concatenate([poses, np.tile(last_row, [poses.shape[0],poses.shape[1],1,1])], axis=2))
        self.intrinsics = torch.Tensor(np.array([[np.array([[hwf[2], 0, hwf[1]/2,0],[0, hwf[2], hwf[0]/2,0],[0, 0, 1,0],[0,0,0,1]]) for hwf in scene] for scene  in hwfs]))
        self.bds = torch.Tensor(poses_bounds[:,:, -2:])
        


    def get_pose_intrinsics(self, scene_indices,image_indices):
        # assert False, f"scene_indices  min {scene_indices.min()} max {scene_indices.max()} image_indices min {image_indices.min()} max {image_indices.max()}"
        return self.poses[scene_indices,image_indices],self.intrinsics[scene_indices,image_indices]
    
    def get_image_encodings(self, image_indices,x,y):
        return self.image_encodings
    def get_video_embedding(self):
        return self.video_embeddings
    def get_video_flows(self):
        return self.video_flow
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):

        #
        #  False, f"index {index} {self.indices[index]}"
        scene_index,image_index,y,x = self.indices[index]
    # scene_index_cuda,image_index_cuda,y_cuda,x_cuda = scene_index.to(device=self.device),image_index.to(device=self.device),y.to(device=self.device),x.to(device=self.device)
        # assert False, f"scene_index {scene_index} image_index {image_index} x {x} y {y}"
        # assert False, f"rays_o {self.rays_o_np.shape} rays_d {self.rays_d_np.shape}"
        rays_o = self.rays_o_np[scene_index,image_index,:,y,x]
        rays_d = self.rays_d_np[scene_index,image_index,:,y,x]
        image_indices = torch.tensor(np.clip([image_index-1,image_index,image_index+1],0,self.n_images-1))
        ind = (scene_index,image_indices,x,y)
        pose = self.poses[scene_index,image_indices]
        intrinsic = self.intrinsics[scene_index,image_indices]
        image_mask = self.images_masks[scene_index,image_index,y,x]
        disparity = self.disparities[scene_index,image_index,y,x]
        rgb = self.images[scene_index,image_index,:,y,x]
        fwd_flow = self.flows_fwd[scene_index,image_index,:,y,x] if image_index< self.n_images-1 else torch.zeros(2).float()
        fwd_flow_mask = self.images_masks[scene_index,image_index,y,x] if image_index<=self.n_images-1 else torch.zeros(1).float().squeeze()
        bwd_flow = self.flows_bwd[scene_index,image_index,:,y,x] if image_index>0 else torch.zeros(2).float()
        bwd_flow_mask = self.images_masks[scene_index,image_index,y,x] if image_index>0 else torch.zeros(1).float().squeeze()
        return ind,rays_o,rays_d,pose,intrinsic,image_mask,disparity,fwd_flow,fwd_flow_mask,bwd_flow,bwd_flow_mask,rgb
        # assert False, f"rgb {rgb} image_encoding {image_encoding.shape} image_mask {image_mask} rays_o {rays_o} rays_d {rays_d}"
        

if __name__ == "__main__":
    # Load the images
    # import torch
    from util.config import *

    config = get_config()
    dataset = CustomEncodingsImageDataset(config.data_folders,config)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
    for a in loader:
        print(a)
        # assert False, f"item.shape {item.shape}"
#     # Print the shape of the images tensor
#     print(images.shape)


# class CustomImageDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         image_path = self.image_paths[index]
#         image = Image.open(image_path).convert('RGB')

#         if self.transform is not None:
#             image = self.transform(image)

#         return image