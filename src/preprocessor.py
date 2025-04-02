import cv2
import numpy as np
import os
import json

class ImagePreprocessor:
    def __init__(self, detection_params_path=None, check_type=None):
        self.check_type = check_type or "default"
        self.preprocessing_settings = {
            "adaptive_threshold": True,
            "clahe": True,
            "denoise": True,
            "sharpen": True
        }
        if detection_params_path and os.path.exists(detection_params_path):
            try:
                with open(detection_params_path, 'r') as f:
                    params = json.load(f)
                if self.check_type in params and "preprocessing" in params[self.check_type]:
                    self.preprocessing_settings = params[self.check_type]["preprocessing"]
            except Exception as e:
                print(f"Error loading preprocessing settings: {e}")

    def load_image(self, image_path):
        return cv2.imread(image_path)

    def to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def apply_threshold(self, image, method='adaptive'):
        if method == 'adaptive':
            return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif method == 'otsu':
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        return image

    def denoise(self, image, strength=10):
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)

    def deskew(self, image):
        moments = cv2.moments(image)
        if moments['mu02'] > 0:
            skew = moments['mu11'] / moments['mu02']
            angle = np.degrees(np.arctan(skew))
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return image

    def enhance_contrast(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def sharpen(self, image):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def preprocess(self, image_path_or_array, deskew=True, denoise_strength=10, threshold_method='adaptive', enhance=True):
        if isinstance(image_path_or_array, str):
            original = self.load_image(image_path_or_array)
        else:
            original = image_path_or_array.copy()
        gray = self.to_grayscale(original)
        processed = gray.copy()
        if enhance and self.preprocessing_settings.get('clahe', True):
            processed = self.enhance_contrast(processed)
        if self.preprocessing_settings.get('adaptive_threshold', True):
            processed = self.apply_threshold(processed, method=threshold_method)
        if denoise_strength > 0 and self.preprocessing_settings.get('denoise', True):
            processed = self.denoise(processed, strength=denoise_strength)
        if deskew:
            processed = self.deskew(processed)
        if self.preprocessing_settings.get('sharpen', True):
            processed = self.sharpen(processed)
        return original, processed