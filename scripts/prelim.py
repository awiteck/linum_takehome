import os
from PIL import Image

# Path to the directory containing images
directory_path = "./validation/corrupted_imgs/"

# Iterate through each file in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a PNG image
    if filename.endswith(".png"):
        # Construct the full path to the image
        img_path = os.path.join(directory_path, filename)
        # Open the image
        with Image.open(img_path) as img:
            # Print the filename and the image size
            print(f"{filename}: {img.size}")  # img.size is a tuple (width, height)
