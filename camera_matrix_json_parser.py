import json
import os
import numpy as np

data = json.load(open("/home/yiftach/main/Research/MonoNeRF/data/pac_nerf_data/bird/all_data.json"))
# images_in_folder = os.listdir("/home/yiftach/main/Research/MonoNeRF/flow_with_background")
numbers = list(range(14))
file_names = [f"./data/r_0_{number}.png" for number in numbers]
def generate_file_name(metadata):
    file_name = metadata["file_path"].split(".")[1].split("/")[-1]
    return f"{file_name}.png"
def extract_frame_number(file_name):
    return int(file_name.split(".")[1].split("/")[-1].split("_")[-1])
file_paths = list([metadata for metadata in data if metadata['file_path'] in file_names])
file_paths = list(sorted(file_paths, key=lambda x: x["time"]))
# print(file_paths)

c2ws = np.array(([np.array(metadata["c2w"]) for metadata in file_paths]))
intrinsics = np.array(([np.array(metadata["intrinsic"]) for metadata in file_paths]))
np.save(open("data/pac_nerf_data/bird/2/c2ws.npy", "wb"), c2ws)
np.save(open("data/pac_nerf_data/bird/2/intrinsics.npy", "wb"), intrinsics)