import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from tqdm import tqdm
from torchvision.io import write_jpeg
import torchvision.transforms as transforms
from dataloader import *

plt.rcParams["savefig.bbox"] = "tight"




import tempfile
from pathlib import Path
from urllib.request import urlretrieve


video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
_ = urlretrieve(video_url, video_path)

from torchvision.io import read_video
# frames, _, _ = read_video(str(video_path), output_format="TCHW",pts_unit="sec")
frames = torch.Tensor(load_images_as_tensor('data/ball/image')).to(device="cuda").permute(1,0,2,3)
# frames = F.interpolate(frames,size=(240,360))
masks = torch.Tensor(load_images_as_tensor('data/ball/mask')).to(device="cuda").permute(1,0,2,3)/255
# assert False, f"frames.shape = {frames.shape} masks.shape = {masks.shape} max mask = {masks.max()}"
# frames = frames*(1-masks)
# foreground_images = frames*(1-masks)
# assert False, f"frames.shape = {frames.shape}"

# assert False, f"frames.shape = {frames.shape}"
img1_batch = torch.stack([frames[100], frames[110]])
img2_batch = torch.stack([frames[19], frames[20]])



from torchvision.models.optical_flow import Raft_Large_Weights

weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()


def preprocess(img1_batch, img2_batch):
    
    
    # img1_batch = F.interpolate(img1_batch, size=(520, 960), mode='bilinear', align_corners=False)
    # img2_batch = F.interpolate(img2_batch, size=(520, 960), mode='bilinear', align_corners=False)
    return transforms(img1_batch, img2_batch)


img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

from torchvision.models.optical_flow import raft_large

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")

predicted_flows = list_of_flows[-1]
print(f"dtype = {predicted_flows.dtype}")
print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")
from torchvision.utils import flow_to_image

# flow_imgs = flow_to_image(predicted_flows)

# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
# ?low_img) in zip(img1_batch, flow_imgs)]
from torchvision.io import write_jpeg
# Determine the number of iterations needed
flows_per_image = []
for i, (img1, img2) in enumerate(zip(frames, frames[1:])):
    with torch.no_grad():
        # Note: it would be faster to predict batches of flows instead of individual flows
        img1, img2 = preprocess(img1, img2)
        img1,img2 = img1.unsqueeze(0),img2.unsqueeze(0)
        # assert False, f"img1.shape = {img1.shape} img2.shape = {img2.shape}"
        list_of_flows = model(img1.to(device), img2.to(device))
        predicted_flow = list_of_flows[-1]
        flows_per_image += [predicted_flow]
flows_per_image = torch.cat(flows_per_image)
torch.save(flows_per_image, 'data/ball/data/flow.pt')
assert False, f"flows_per_image.shape = {flows_per_image.shape}"