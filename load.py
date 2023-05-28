import torch
import os
from tqdm import tqdm 
directory_out = '/home/yiftach/main/Research/MonoNeRF/data/ball/data'  # Replace with your directory path
directory = '/home/yiftach/main/Research/MonoNeRF/data/ball/image_encodings'  # Replace with your directory path

concatenated_tensors = None
files = os.listdir(directory)
sorted_files = sorted(files, key=lambda x: int(x.split("_")[2]))
# assert False, f"os.listdir(directory) {sorted_files}"
# Iterate over the files in the directory
for filename in tqdm(sorted_files):
    with torch.no_grad():
        if filename.endswith('.pt'):
            filepath = os.path.join(directory, filename)
            
            # Load the tensor from each file
            tensor = torch.load(filepath).to(device="cpu")
            
            # Concatenate the tensors
            if concatenated_tensors is None:
                concatenated_tensors = tensor
            else:
                concatenated_tensors = torch.cat((concatenated_tensors, tensor), dim=0)

# Save the concatenated tensors to a new pt file
output_path = os.path.join(directory_out, 'image_encoding.pt')
torch.save(concatenated_tensors, output_path)
