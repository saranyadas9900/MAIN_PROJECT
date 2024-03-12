import os
import time
import cv2
import numpy as np
from training import AppraisalVisionTrainer  # Import your trainer class from the training file

# Load the trained model
trainer = AppraisalVisionTrainer()
trainer.load_model("rf_model_max.pkl")

# Paths to your floorplan and non-floorplan image folders
floorplan_folder = "floorplan_images"
non_floorplan_folder = "nonfloorplan_images"

# Combine images from both folders
all_images = []
for folder in [floorplan_folder, non_floorplan_folder]:
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        all_images.append(image_path)

# Start the timer
start_time = time.time()

# Make predictions for each image
predictions = []
for image_path in all_images:
    # Read the image
    image = cv2.imread(image_path)

    # Extract features if necessary
    features = trainer.extract_features(image)

    # Reshape features if necessary
    reshaped_features = np.array(features).reshape(1, -1)  # Reshaping to match the model's input format

    # Make predictions
    prediction = trainer.model.predict(reshaped_features)
    predictions.append(prediction)

# End the timer
end_time = time.time()

# Calculate the total inference time
inference_time = end_time - start_time
print("Total Inference Time:", inference_time, "seconds")
