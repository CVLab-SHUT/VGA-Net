# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

# Function to extract random patches from an image
def extract_random_patches(image, patch_size, num_patches):
    patches = []
    height, width, _ = image.shape
    for _ in range(num_patches):
        # Generate random coordinates for the top-left corner of the patch
        top_left_x = np.random.randint(0, width - patch_size)
        top_left_y = np.random.randint(0, height - patch_size)
        
        # Extract the patch from the image
        patch = image[top_left_y:top_left_y+patch_size, top_left_x:top_left_x+patch_size]
        patches.append(patch)
    return patches

# Directory containing preprocessed images
preprocessed_dir = 'yeganeh/preprocessed/images'

# Directory to save extracted patches
patches_dir = 'yeganeh/extracted/patches'

# Create the patches directory if it doesn't exist
if not os.path.exists(patches_dir):
    os.makedirs(patches_dir)

# Patch size
patch_size = 48

# Number of patches to extract from each image
num_patches_per_image = 143

# List all preprocessed images in the preprocessed directory
file_list = os.listdir(preprocessed_dir)

# Iterate over each preprocessed image
for file_name in file_list:
    # Read the preprocessed image
    preprocessed_image = cv2.imread(os.path.join(preprocessed_dir, file_name))
    
    # Extract random patches from the preprocessed image
    patches = extract_random_patches(preprocessed_image, patch_size, num_patches_per_image)
    
    # Save the extracted patches
    for i, patch in enumerate(patches):
        patch_name = os.path.splitext(file_name)[0] + f'_patch_{i}.jpg'
        cv2.imwrite(os.path.join(patches_dir, patch_name), patch)

