import torch
from tqdm import tqdm
images = torch.load("/home/yiftach/main/Research/MonoNeRF/data/Balloon1/embeddings/image_embeddings_static.pt").cpu()
print(images.shape)
import torch
from sklearn.decomposition import PCA

# Assuming input is your feature maps tensor of shape [12, 832, 270, 480]
def apply_pca(input, k):
    # Get the shape of the input tensor
    original_size = input.size()

    # Reshape the tensor to be 2D for PCA
    reshaped = input.view(original_size[0], original_size[1], -1)
    
    # Initialize PCA
    pca = PCA(n_components=k)

    # Apply PCA to each feature map
    output = []
    for i in tqdm(range(reshaped.size(0))):
        output_i = pca.fit_transform(reshaped[i].numpy().T)
        output.append(torch.from_numpy(output_i.T))

    # Stack all the output tensors together
    output = torch.stack(output)

    # Reshape the output tensor to match the original spatial dimensions
    output = output.view(original_size[0], k, original_size[2], original_size[3])

    return output

# The number of principal components you want to keep
K = 256

# Apply PCA
output = apply_pca(images, K)
torch.save(output, "/home/yiftach/main/Research/MonoNeRF/data/Balloon1/embeddings/image_embeddings_static_pca.pt")