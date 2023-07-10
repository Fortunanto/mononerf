import torch
from carvekit.api.high import HiInterface
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Check doc strings for more information
interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)
images = list(sorted(os.listdir("/home/yiftach/main/Research/MonoNeRF/data/ball/image")))
# assert False, f"images = {images}"
images = [f'/home/yiftach/main/Research/MonoNeRF/data/ball/image/{image}' for image in images]

images = interface(images)
for i,image in tqdm(enumerate(images)):
    # img = Image.open(os.path.join(input_directory, filename))

    # Convert the image data to numpy array for manipulation
    data = np.array(image)

    # Create an empty array with the same shape as our image data
    mask = np.zeros(data.shape, dtype=np.uint8)

    # Set the alpha channel values in your mask to 255 where the original image alpha values are > 0
    mask[data[:, :, 3] > 0] = 255

    # Create a new image from the mask
    mask_img = Image.fromarray(mask)

    # Save the mask image
    mask_img.save(os.path.join("/home/yiftach/main/Research/MonoNeRF/data/ball/mask","{:04d}.png".format(i)))
    
    # mask.save(f'temp/{i}_mask.png')
# assert False, f"len(images_without_background) = {len(images_without_background)}"
# cat_wo_bg = images_without_background[0][0]

# cat_wo_bg.save('2.png')
# cat_wo_bg_mask = images_without_background[1][0]
# cat_wo_bg_mask.save('2_mask.png')