import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define paths to the directories containing floor plan and non-floor plan images
floorplan_dir = "floor"
non_floorplan_dir = "homes"

# Function to load and display a random sample of images from a directory
def display_random_images(image_dir, num_images=5):
    image_files = os.listdir(image_dir)
    random_sample = np.random.choice(image_files, size=num_images, replace=False)
    
    fig, axs = plt.subplots(1, num_images, figsize=(15, 3))
    fig.suptitle("Random Sample of Images")
    
    for i, image_file in enumerate(random_sample):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[i].imshow(image)
        axs[i].axis("off")
        axs[i].set_title(image_file)

# Function to calculate basic image statistics
def calculate_image_statistics(image):
    mean_pixel_value = np.mean(image)
    std_dev_pixel_value = np.std(image)
    min_pixel_value = np.min(image)
    max_pixel_value = np.max(image)
    
    return mean_pixel_value, std_dev_pixel_value, min_pixel_value, max_pixel_value

# Function to plot color histograms
def plot_color_histogram(image, title):
    colors = ("r", "g", "b")
    plt.figure(figsize=(8, 5))
    plt.title(title)
    
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histogram, color=color)
    
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

# Function to perform Canny edge detection
def perform_canny_edge_detection(image, title):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)  # Adjust thresholds as needed
    
    plt.figure(figsize=(8, 5))
    plt.imshow(edges, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

# Display random sample of floor plan and non-floor plan images
display_random_images(floorplan_dir, num_images=5)
display_random_images(non_floorplan_dir, num_images=5)

# Calculate image statistics for a sample image
sample_image_path = os.path.join(floorplan_dir, os.listdir(floorplan_dir)[0])
sample_image = cv2.imread(sample_image_path)
mean_pixel, std_dev_pixel, min_pixel, max_pixel = calculate_image_statistics(sample_image)
print("Image Statistics:")
print(f"Source Directory: {floorplan_dir}") 
print(f"Mean Pixel Value: {mean_pixel}")
print(f"Standard Deviation of Pixel Value: {std_dev_pixel}")
print(f"Minimum Pixel Value: {min_pixel}")
print(f"Maximum Pixel Value: {max_pixel}")

# Plot color histogram for the sample image
plot_color_histogram(sample_image, title="Color Histogram")

# Perform Canny edge detection for the sample image
perform_canny_edge_detection(sample_image, title="Canny Edge Detection")
