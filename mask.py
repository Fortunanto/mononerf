import cv2
import os
import numpy as np 
from PIL import Image

input_directory = "temp"
output_directory = "temp_mask"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

lower_white = np.array([200, 200, 200], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)

for filename in os.listdir(input_directory):
    # # assert False, f"filename = {filename}"
    # if filename.endswith(".jpg") or filename.endswith(".png"):
    #     # read the image
    #     img = cv2.imread(os.path.join(input_directory, filename))

    #     # find white color in the image
    #     white_mask = cv2.inRange(img, lower_white, upper_white)

    #     # invert the mask to get the object
    #     object_mask = cv2.bitwise_not(white_mask)

    #     # save the binary image
    #     im_pil = Image.fromarray(object_mask)
    #     im_pil.save(os.path.join(output_directory, f"mask_{filename}"))
    # # Open the image file
    img = Image.open(os.path.join(input_directory, filename))

    # Convert the image data to numpy array for manipulation
    data = np.array(img)

    # Create an empty array with the same shape as our image data
    mask = np.zeros(data.shape, dtype=np.uint8)

    # Set the alpha channel values in your mask to 255 where the original image alpha values are > 0
    mask[data[:, :, 3] > 0] = 255

    # Create a new image from the mask
    mask_img = Image.fromarray(mask)

    # Save the mask image
    mask_img.save(os.path.join(output_directory,f'mask_{filename}'))