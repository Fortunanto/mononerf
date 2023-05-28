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

slowfast_alpha = 4
create_compose = lambda mean, std, crop_size,side_size: Compose( [
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ])


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        frame_count = frames.shape[1]
        batch_size = frame_count//32
        frames = frames[:,0:batch_size*32,:,:].reshape(batch_size,frames.shape[0],32,frames.shape[2],frames.shape[3])
        fast_pathway = frames
        frames = frames.to(device="cpu")
        # assert False, f"frames.shape {frames.shape}, frame_count {frame_count}"
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            2,
            torch.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway.to(device="cuda"), fast_pathway]
        return frame_list


    # Custom dataset to load images from a directory without class-wise subdirectories
    class CustomImageDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert('RGB')

            if self.transform is not None:
                image = self.transform(image)

            return image



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
    images_tensor = np.stack(images).transpose(3,0,1,2)
    
    return images_tensor

def prepare_for_slowfast(tensor, mean, std, crop_size,device="cpu"):
    transforms = create_compose(mean, std, crop_size,400)
    [slow,fast] = transforms(tensor)
    [slow,fast] = [slow.to(device),fast.to(device)]
    # tensor = tensor.reshape(tensor.shape[0],tensor.shape[1]*tensor.shape[2],tensor.shape[3],tensor.shape[4])
    return [slow,fast]

def map_to_indices(index,shape):
    a,_,c,d = shape
    a_index = index // (c * d)
    rem = index % (c * d)
    c_index = rem // d
    d_index = rem % d

    indices = (a_index, c_index, d_index)
    assert False, f"index {index}, indices {indices}, shape {shape}"

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    
    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose
    
import itertools
class CustomEncodingsImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir, transform=None):
        self.data_dir = os.path.join(dir,"data")
        self.image_dir = os.path.join(dir,"image") 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
                  
        self.images = torch.Tensor(load_images_as_tensor(self.image_dir).transpose(1,0,2,3))
        self.images = F.interpolate(self.images,size=(240,360))
        self.image_encodings = torch.load(os.path.join(self.data_dir, "image_encoding.pt"))
        self.video_embeddings = torch.load(os.path.join(self.data_dir, "video_embedding.pt"))
        self.video_flow = torch.zeros((150,2,240,360)).to(device=self.device)
        h,w=240,360
        self.n_images = self.images.shape[0]
        self.intrinsics,self.poses = torch.load(os.path.join(self.data_dir, "intrinsics.pt")),torch.load(os.path.join(self.data_dir, "pose.pt"))
        self.intrinsics.requires_grad = False
        self.poses.requires_grad = False
        # self.indices = list(itertools.product(range(1,self.n_images-1),range(h),range(w)))
        self.indices = list(itertools.product(range(1,3),range(2),range(2)))


        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])

        self.object_bbox_min = object_bbox_min[:, None]
        self.object_bbox_max = object_bbox_max[:, None] 
    def get_pose_intrinsics(self, image_indices):
        return self.poses[image_indices],self.intrinsics[image_indices]
    
    def get_image_features(self, image_indices,x,y):
        return self.image_encodings[image_indices,:,x,y]
    def get_video_embedding(self):
        return self.video_embeddings
    def get_video_flows(self):
        return self.video_flow
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        # assert False, f"index {index} {self.indices[index]}"
        return self.indices[index],self.images[self.indices[index][0],:,self.indices[index][1],self.indices[index][2]]

        

if __name__ == "__main__":
    # Load the images
    # import torch

    shape = (3, 4, 5)
    lower_bounds = [0, 10, 20]
    upper_bounds = [9, 19, 29]

    random_tensor = torch.randint(0, 100, shape)  # Generate random integers in a large range

    # Apply masks to restrict the range per dimension
    for dim, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
        if lower != -1 or upper != -1:
            mask = ((random_tensor >= lower) & (random_tensor <= upper)).float()
            random_tensor = random_tensor * mask

    print(random_tensor)

    dataset = CustomEncodingsImageDataset("data/ball")
    get_image_features = dataset.get_image_features([19,20,21],[0,1,2],[0,1,2])
    print(get_image_features.shape)
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