import cv2
import numpy as np
from training1 import AppraisalVisionTrainer  # Import your trainer class from the training file

def predict_floor_plan(image_path):
    # Load the trained model
    trainer = AppraisalVisionTrainer()
    trainer.load_model("rf_model_max.pkl")

    # Read the input image
    image = cv2.imread(image_path)

    # Resize the input image
    resized_image = trainer.resize_image(image)

    # Extract features
    features = trainer.extract_features(resized_image)

    # Define feature names
    feature_names = [
        "Saturation",
        "Dominant Color Extraction",
        "Color Palette Extraction",
        "Lines Count",
        "Contour Count",
        "Laplacian Variance"
    ]

    # Reshape features if necessary
    reshaped_features = np.array(features).reshape(1, -1)  # Reshaping to match the model's input format

    # Predict whether the image is a floor plan or not
    prediction = trainer.model.predict(reshaped_features)

    # Interpret the prediction
    result = ""
    if prediction == 1:
        result = "Predicted image: floor plan."
    else:
        result = "Predicted image: non-floor plan."

    # Print each feature name and value
    for feature_name, feature_value in zip(feature_names, features):
        print(f"{feature_name}: {feature_value}")

    return result

if __name__ == "__main__":
    # Path to the image you want to classify
    image_path_to_classify = "path_to_your_image.jpg"

    # Perform prediction to get the result
    result = predict_floor_plan(image_path_to_classify)
    print(result)
