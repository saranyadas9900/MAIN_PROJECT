import cv2
import numpy as np
import pickle
class AppraisalVision:
    def __init__(self):
        self.model = pickle.load(open("rf_model_max.pkl","rb"))
    def saturation_check(self,image):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:, :, 1].mean()
        return saturation

    def dominant_color_extraction(self,image):
        colors, counts = np.unique(image.reshape(-1, 1), axis=0, return_counts=True)
        index = list(counts).index(max(counts))
        return colors[index][0]
    def color_pellete_extraction(self,image):
        colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
        return len(colors)
    def lines_count(self,image):

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
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
        if lines is None or len(lines)==0:
            return 0
        else:
            return len(lines)

    def contour_count(self,image):

        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray=255-gray
        kernel = np.array([[0,-1,0], [0,5,0], [0,-1,0]])
        gray = cv2.filter2D(gray, -1, kernel)
        contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST ,cv2.CHAIN_APPROX_NONE )
        if contours is None or len(contours)==0:
            return 0
        else:
            return len(contours)

    def process_laplacian(self,image):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = np.asarray(image)
        image = image / 255.0
        laplacian= cv2.Laplacian(image, cv2.CV_64F)
        return laplacian.max(),laplacian.var()
    def floor_plan_classifier(self,image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (500, 500))
        features= [self.saturation_check(image),self.dominant_color_extraction(image),
                   self.color_pellete_extraction(image),self.lines_count(image),
                   self.contour_count(image),self.process_laplacian(image)[1]]
        features=np.array(features)
        features=features.reshape(1,-1)
        return self.model.predict(features)[0]