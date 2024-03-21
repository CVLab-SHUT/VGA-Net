# -*- coding: utf-8 -*-

#-------------------------------
#pre_process v1 (with replace_black_area)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to replace black areas in an image
def replace_black_area(image):
    # Convert the image to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate the average color of each channel
    average_red = np.mean(image[:, :, 0])
    average_green = np.mean(image[:, :, 1])
    average_blue = np.mean(image[:, :, 2])

    # Create a mask of the black area
    black_mask = np.all(image < [average_red, average_green, average_blue], axis=2)

    # Replace the black area with the average color
    image[black_mask] = [average_red, average_green, average_blue]

    # Blur the image to smooth out the edges
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert the image back to BGR color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Return the modified image
    return image

# Function to apply CLAHE to enhance image contrast
def apply_clahe(image):
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_l_channel = clahe.apply(l_channel)

    # Merge the CLAHE-enhanced L channel with the original A and B channels
    clahe_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))

    # Convert the CLAHE-enhanced LAB image back to BGR color space
    clahe_bgr_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)

    return clahe_bgr_image

# Function to apply unsharp masking to sharpen the image
def unsharp_mask(image, sigma=1.0, strength=1.5):
    # Split the image into B, G, and R channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Apply Gaussian blur to each channel
    blurred_b = cv2.GaussianBlur(b_channel, (0, 0), sigma)
    blurred_g = cv2.GaussianBlur(g_channel, (0, 0), sigma)
    blurred_r = cv2.GaussianBlur(r_channel, (0, 0), sigma)

    # Calculate the sharpened image for each channel
    sharp_b = cv2.addWeighted(b_channel, 1.0 + strength, blurred_b, -strength, 0)
    sharp_g = cv2.addWeighted(g_channel, 1.0 + strength, blurred_g, -strength, 0)
    sharp_r = cv2.addWeighted(r_channel, 1.0 + strength, blurred_r, -strength, 0)

    # Merge the sharpened channels into a single image
    sharp_image = cv2.merge((sharp_b, sharp_g, sharp_r))

    return sharp_image

# Function to replace black areas back in the original image
def replace_black_area_back(original_image, modified_image):
    # Convert the images to RGB color space
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)

    # Calculate the average color of each channel for the original image
    average_red_original = np.mean(original_image[:, :, 0])
    average_green_original = np.mean(original_image[:, :, 1])
    average_blue_original = np.mean(original_image[:, :, 2])

    # Create a mask of the black area for the modified image
    black_mask = np.all(modified_image < [average_red_original, average_green_original, average_blue_original], axis=2)

    # Replace the black area in the modified image with the original values
    modified_image[black_mask] = original_image[black_mask]

    # Convert the image back to BGR color space
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR)

    # Return the modified image with black areas restored
    return modified_image

# Load the original image
original_image = cv2.imread('23_training.jpg')

# Apply each step sequentially
modified_image = replace_black_area(original_image)
clahe_image = apply_clahe(modified_image)
unsharp_masked_image = unsharp_mask(clahe_image)
restored_image = replace_black_area_back(original_image, unsharp_masked_image)

# Display all the images
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')

axs[0, 1].imshow(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Image with Black Area Replaced')

axs[0, 2].imshow(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2RGB))
axs[0, 2].set_title('CLAHE Enhanced Image')

axs[1, 0].imshow(cv2.cvtColor(unsharp_masked_image, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Unsharp Masked Image')

axs[1, 1].imshow(cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Restored Image')

# Hide the axes
for ax in axs.flat:
    ax.axis('off')

# Display the images
plt.tight_layout()
plt.show()


#----------------------------
#pre_process v2 (without replace_black_area)
# Load the original image
original_image = cv2.imread('23_training.jpg')

# Convert the image to LAB color space
lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)

# Split the LAB image into L, A, and B channels
l_channel, a_channel, b_channel = cv2.split(lab_image)

# Apply CLAHE to the L channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_l_channel = clahe.apply(l_channel)

# Merge the CLAHE-enhanced L channel with the original A and B channels
clahe_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))

# Convert the CLAHE-enhanced LAB image back to BGR color space
clahe_bgr_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)

# Display the original and CLAHE-enhanced images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(clahe_bgr_image, cv2.COLOR_BGR2RGB))
plt.title('CLAHE Enhanced Image')
plt.axis('off')

# Function to apply unsharp masking to sharpen the image
def unsharp_mask(image, sigma=1.0, strength=1.5):
    # Split the image into B, G, and R channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Apply Gaussian blur to each channel
    blurred_b = cv2.GaussianBlur(b_channel, (0, 0), sigma)
    blurred_g = cv2.GaussianBlur(g_channel, (0, 0), sigma)
    blurred_r = cv2.GaussianBlur(r_channel, (0, 0), sigma)

    # Calculate the sharpened image for each channel
    sharp_b = cv2.addWeighted(b_channel, 1.0 + strength, blurred_b, -strength, 0)
    sharp_g = cv2.addWeighted(g_channel, 1.0 + strength, blurred_g, -strength, 0)
    sharp_r = cv2.addWeighted(r_channel, 1.0 + strength, blurred_r, -strength, 0)

    # Merge the sharpened channels into a single image
    sharp_image = cv2.merge((sharp_b, sharp_g, sharp_r))

    return sharp_image

# Apply unsharp masking to the CLAHE-enhanced image
unsharp_masked_image = unsharp_mask(clahe_bgr_image)

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(unsharp_masked_image, cv2.COLOR_BGR2RGB))
plt.title('Unsharp Masked Image')
plt.axis('off')
plt.tight_layout()
plt.show()
#----------------------------
#save_results
import os

# Directory containing your dataset
dataset_dir = 'yeganeh/dataset/DRIVE'

# Directory to save preprocessed images
output_dir = 'yeganeh/extracted/patches'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all files in the dataset directory
file_list = os.listdir(dataset_dir)

# Iterate over each file in the dataset directory
for file_name in file_list:
    # Read the original image
    original_image = cv2.imread(os.path.join(dataset_dir, file_name))
    
    # Apply each step of preprocessing sequentially
    modified_image = replace_black_area(original_image)
    clahe_image = apply_clahe(modified_image)
    unsharp_masked_image = unsharp_mask(clahe_image)
    restored_image = replace_black_area_back(original_image, unsharp_masked_image)
    
    # Save the preprocessed image
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, restored_image)
