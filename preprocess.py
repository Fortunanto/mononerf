import argparse
from architecture.encoders import *
from architecture.residual_mlp import *
from dataloader import *
import torch
import torch.nn as nn
# from architecture.implicit_velocity_field import ImplicitVelocityField
from tqdm import tqdm
import shutil
import os
from torchvision import transforms
import torch.nn.functional as F
from architecture.encoder import ImageEncoder
# Path: train.py

def train(model, train_loader, optimizer, criterion, device):
    pass

def main( data_path, output_dir, extract_video_embeddings):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # assert False, f"device: {device}"
    if extract_video_embeddings:
        model = SlowfastVideoEncoder(in_dim=10, out_dim=10).to(device=device).eval()
    else:
        model = ImageEncoder().to(device=device).eval()

    for dir in tqdm(["data/Balloon1", "data/Balloon2","data/Umbrella","data/Truck","data/Skating","data/Playground","data/Jumping"]):
        with torch.no_grad():
            path = os.path.join(dir, "images")
            images = torch.Tensor(load_images_as_tensor(path)).to(device=device)
            # assert False, f"images shape {images.shape}"
            if extract_video_embeddings:
                # Prepare images for video embeddings
                slowfast_data = prepare_for_slowfast(images, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
                                                     , 400,
                                                     device=device)
                assert False, f"slowfast_data[0].shape: {slowfast_data[0].shape} slowfast_data[1].shape: {slowfast_data[1].shape}"
                output = model(slowfast_data)
                assert False, f"output.shape: {output.shape}"
                file_name = "video_embedding.pt"
                path_processed = os.path.join(data_path, dir, output_dir)
                path_output = os.path.join(path_processed, file_name)
                torch.save(output, path_output)


            else:
                images=images.permute(1, 0, 2, 3)
                file_name = "image_embeddings_static.pt"
                preprocess = transforms.Compose([
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                # assert False, f"images.shape: {images.shape}"
                output = []
                for i,batch in enumerate(range(0,images.shape[0],16)):
                    images_output =preprocess(images[batch:batch+16,...])
                    # assert False, f"images_output.shape: {images_output.shape}"
                    output = model(images_output)
                    print("output.shape",output.shape)
                    os.makedirs(os.path.join(dir),exist_ok=True)
                    torch.save(output, os.path.join(dir,'embeddings', f"image_embeddings_static.pt"))
                
                # assert False, f"images.shape: {images.shape}"
                

            # Create processed directory
            # if not os.path.exists(path_processed):
            #     os.mkdir(path_processed)
            # elif os.path.exists(path_processed) and len(os.listdir(path_processed)) != 1 :
            #     shutil.rmtree(path_processed)
            #     os.mkdir(path_processed)
            # Save output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--extract_video_embeddings", action="store_true", help="Flag to extract video embeddings")

    args = parser.parse_args()

    main(args.data_path, args.output_dir, args.extract_video_embeddings)
