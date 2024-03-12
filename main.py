import os
import cv2
import numpy as np


# Function to preprocess a single image
def preprocess_image(image):
    # Resize image to a common scale
    image_resized = cv2.resize(image, (500, 500))

    # Convert image to HSV color space
    image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    # Normalize pixel values
    image_normalized = image_hsv.astype('float32') / 255.0

    return image_normalized


# Function to preprocess all images in a folder
def preprocess_images_in_folder(folder_path):
    # Create a new directory to save preprocessed images
    output_folder = folder_path + "_preprocessed"
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each image in the folder
    for filename in os.listdir(folder_path):
        # Read image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Preprocess image
        preprocessed_image = preprocess_image(image)

        # Save preprocessed image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, (preprocessed_image * 255).astype(np.uint8))

    print(f"Preprocessing completed for images in '{folder_path}' folder.")


# Paths to folders containing images
floor_folder = "floor"
homes_folder = "homes"

# Preprocess images in the "floor" folder
preprocess_images_in_folder(floor_folder)

# Preprocess images in the "homes" folder
preprocess_images_in_folder(homes_folder)
