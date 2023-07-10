from PIL import Image
import numpy as np
import imageio

def rgba_to_rgb(image_path):
 
    # Open the image file
    img = Image.open(image_path).convert('L')  # Convert image to grayscale

    # Convert image data to a numpy array (it's faster to process than PIL's Image class)
    img_array = np.array(img)

    # Choose a threshold (middle of the grayscale range seems a good choice)
    threshold = 128  

    # Apply threshold. This will create a boolean mask where True values correspond to pixels above threshold.
    binary_image_array = img_array > threshold

    # Convert boolean array back to uint8
    binary_image_array = binary_image_array.astype(np.uint8)

    # Multiply by 255 (because True is equivalent to 1, and we want those pixels to be bright (255))
    binary_image_array *= 255

    # Convert back to Image and save
    binary_image = Image.fromarray(binary_image_array)
    binary_image.save('b.png')

# Usage example
rgba_to_rgb('/home/yiftach/main/Research/MonoNeRF/data/ball/mask/0000.png')
img_arr = Image.open('b.png')
img_arr = np.array(img_arr)
assert False, f'img {img_arr.shape}'