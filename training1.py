import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Suppress libpng warnings
with open(os.devnull, 'w') as f:
    old_stderr = os.dup(2)
    os.dup2(f.fileno(), 2)

# Your existing code for the AppraisalVisionTrainer class here...
class AppraisalVisionTrainer:
    def __init__(self):
        self.model = None
        self.target_size = (500, 500)  # Define target size for resizing images

    def resize_image(self, image):
        return cv2.resize(image, self.target_size)

    def saturation_check(self, image):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:, :, 1].mean()
        return saturation

    def dominant_color_extraction(self, image):
        colors, counts = np.unique(image.reshape(-1, 1), axis=0, return_counts=True)
        index = list(counts).index(max(counts))
        return colors[index][0]

    def color_pellete_extraction(self, image):
        colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
        return len(colors)

    def lines_count(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        rho = 1
        theta = np.pi / 180
        threshold = 15
        min_line_length = 50
        max_line_gap = 20
        line_image = np.copy(image) * 0
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        if lines is None or len(lines) == 0:
            return 0
        else:
            return len(lines)

    def contour_count(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
        kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if contours is None or len(contours) == 0:
            return 0
        else:
            return len(contours)

    def process_laplacian(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.asarray(image)
        image = image / 255.0
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return laplacian.max(), laplacian.var()

    def extract_features(self, image):
        # Preprocessing: Resize image
        resized_image = self.resize_image(image)

        features = [
            self.saturation_check(resized_image),
            self.dominant_color_extraction(resized_image),
            self.color_pellete_extraction(resized_image),
            self.lines_count(resized_image),
            self.contour_count(resized_image),
            self.process_laplacian(resized_image)[1]
        ]
        return features
    def evaluate_model(self, X_test, y_test):
        # Make predictions on test data
        y_pred = self.model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Print the evaluation metrics
        print("Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Return evaluation metrics if needed
        return accuracy, precision, recall, f1
    
    def save_model(self, filename):
        # Save the trained model using pickle
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Model saved as {filename}")
    def load_model(self, filename):
        self.model = pickle.load(open(filename, 'rb'))

if __name__ == "__main__":
    trainer = AppraisalVisionTrainer()

    # Train the model
    floorplan_folder = "floor"
    non_floorplan_folder = "homes"
    batch_size = 3000

    # Load and process images from the folders
    X, y = [], []

    # Load floor plan images
    print("Loading floor plan images...")
    for filename in os.listdir(floorplan_folder):
        image_path = os.path.join(floorplan_folder, filename)
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                features = trainer.extract_features(image)
                X.append(features)
                y.append(1)  # Floorplan
                print(f"Loaded floor plan image: {filename}")

    # Load non-floor plan images
    print("Loading non-floor plan images...")
    for filename in os.listdir(non_floorplan_folder):
        image_path = os.path.join(non_floorplan_folder, filename)
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                features = trainer.extract_features(image)
                X.append(features)
                y.append(0)  # Non-floorplan
                print(f"Loaded non-floor plan image: {filename}")

    print("Loaded images for training.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print the number of training and testing data points
    print("Number of training data points:", len(X_train))
    print("Number of testing data points:", len(X_test))

    # Train the model
    trainer.model = RandomForestClassifier(n_estimators=100, random_state=42)
    trainer.model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    trainer.evaluate_model(X_test, y_test)

    # Save the trained model
    trainer.save_model("rf_model_max.pkl")
    

# Restore stderr
os.dup2(old_stderr, 2)
