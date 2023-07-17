import torch
data_bird = torch.load("/home/yiftach/main/Research/PAC-NeRF/data/bird/data.pt")
masks = data_bird["ray_mask_all"].reshape(-1,800,800)
for i,mask in enumerate(masks):
    #write mask as binary image to disk
    from PIL import Image
    image = Image.fromarray(mask.numpy().astype("uint8")*255)
    image.save("mask_%d.png" % i)
    print(image)