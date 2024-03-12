import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    def train(self, floorplan_folder, non_floorplan_folder, batch_size=3000):
        X, y = [], []

        # Load floor plan images
        print("Loading floor plan images...")
        try:
            for filename in os.listdir(floorplan_folder):
                image_path = os.path.join(floorplan_folder, filename)
                if os.path.isfile(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        features = self.extract_features(image)
                        X.append(features)
                        y.append(1)  # Floorplan
                        print(f"Loaded floor plan image: {filename}")
        except Exception as e:
            print(f"Error loading floor plan images: {e}")

        print(f"Loaded {len(X)} floor plan images.")

        # Load non-floor plan images
        print("Loading non-floor plan images...")
        try:
            for filename in os.listdir(non_floorplan_folder):
                image_path = os.path.join(non_floorplan_folder, filename)
                if os.path.isfile(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        features = self.extract_features(image)
                        X.append(features)
                        y.append(0)  # Non-floorplan
                        print(f"Loaded non floor plan image: {filename}")
        except Exception as e:
            print(f"Error loading non-floor plan images: {e}")

        print(f"Loaded {len(X) - len(floorplan_folder)} non-floor plan images.")

        print("Training Random Forest classifier...")
        # Train Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        print("Training complete.")




    def floor_plan_classifier(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (500, 500))
        features = self.extract_features(image)
        return self.model.predict(features)[0]

    def save_model(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))

    def load_model(self, filename):
        self.model = pickle.load(open(filename, 'rb'))


# Redirect standard error to NUL
with open(os.devnull, 'w') as f:
    old_stderr = os.dup(2)
    os.dup2(f.fileno(), 2)
    try:
        # Example usage:
        if __name__ == "__main__":
            trainer = AppraisalVisionTrainer()

            # Train the model
            floorplan_folder = "floor"
            non_floorplan_folder = "homes"
            batch_size = 3000
            trainer.train(floorplan_folder, non_floorplan_folder, batch_size)

            # Save the trained model
            trainer.save_model("rf_model_max.pkl")
    finally:
        os.dup2(old_stderr, 2)

