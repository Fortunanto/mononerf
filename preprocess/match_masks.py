import os
import argparse
import shutil
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to the data directory")
parser.add_argument("--mask_path", type=str, help="path to masks")

args = parser.parse_args()
path = os.path.join(args.data_path,"images")
images = os.listdir(path)
masks = os.listdir(args.mask_path)
masks_dict_to_path = {}
shutil.rmtree(os.path.join(args.data_path,"disp"))

for mask in masks:
    #find image corresponding to mask
    keys = mask.split(".")[0].split("_")
    image_name = f"{keys[2]}_{keys[1]}.png"
    output_name = f"{keys[2]}_{keys[1]}.npy"
    # mask_name = f"{keys[2]}_{keys[1]}_m.png"
    if image_name in images:
        #copy mask to image folder under masks subfolder
        os.makedirs(os.path.join(args.data_path,"disp"),exist_ok=True)
        shutil.copy(os.path.join(args.mask_path,mask),os.path.join(args.data_path,"disp",output_name))
# images = os.listdir(images_path)
print(args)