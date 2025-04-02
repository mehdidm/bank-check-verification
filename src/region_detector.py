import cv2
import numpy as np
import json
import os
from sklearn.cluster import KMeans

class RegionDetector:
    def __init__(self, config_path=None, detection_params_path=None, check_type=None):
        self.config_path = config_path
        self.detection_params_path = detection_params_path
        self.check_type = check_type or "default"
        self.regions_config = self._load_regions_config()
        self.detection_params = self._load_detection_params()

    def _load_regions_config(self):
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {
            'micr_line': {'x1': 0.05, 'x2': 0.95, 'y1': 0.85, 'y2': 0.95},
            'amount_box': {'x1': 0.75, 'x2': 0.95, 'y1': 0.15, 'y2': 0.25},
            'payee_line': {'x1': 0.05, 'x2': 0.75, 'y1': 0.15, 'y2': 0.25},
            'date_line': {'x1': 0.75, 'x2': 0.95, 'y1': 0.05, 'y2': 0.15},
            'written_amount': {'x1': 0.05, 'x2': 0.75, 'y1': 0.35, 'y2': 0.45}
        }

    def _load_detection_params(self):
        if self.detection_params_path and os.path.exists(self.detection_params_path):
            with open(self.detection_params_path, 'r') as f:
                params = json.load(f)
            return params.get(self.check_type, {})
        return {}

    def _detect_contours(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _classify_regions(self, contours, image_shape):
        h, w = image_shape
        features = []
        for contour in contours:
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            features.append([x/w, y/h, w_contour/w, h_contour/h, area/(w*h)])
        
        if not features:
            return {}

        kmeans = KMeans(n_clusters=min(5, len(features)), random_state=42)
        labels = kmeans.fit_predict(features)
        
        regions = {}
        for i, (contour, label) in enumerate(zip(contours, labels)):
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            if label == 0:  # MICR line (bottom)
                regions['micr_line'] = (x, y, x + w_contour, y + h_contour)
            elif label == 1:  # Amount box (top-right)
                regions['amount_box'] = (x, y, x + w_contour, y + h_contour)
            elif label == 2:  # Payee line (left-middle)
                regions['payee_line'] = (x, y, x + w_contour, y + h_contour)
            elif label == 3:  # Date line (top-right)
                regions['date_line'] = (x, y, x + w_contour, y + h_contour)
            elif label == 4:  # Written amount (middle)
                regions['written_amount'] = (x, y, x + w_contour, y + h_contour)
        return regions

    def extract_regions(self, image, method='dynamic'):
        h, w = image.shape[:2]
        regions = {}
        
        if method == 'fixed':
            for name, coords in self.regions_config.items():
                x1, x2 = int(w * coords['x1']), int(w * coords['x2'])
                y1, y2 = int(h * coords['y1']), int(h * coords['y2'])
                regions[name] = image[y1:y2, x1:x2]
        else:  # dynamic
            contours = self._detect_contours(image)
            detected_regions = self._classify_regions(contours, (h, w))
            for name in self.regions_config:
                if name in detected_regions:
                    x1, y1, x2, y2 = detected_regions[name]
                    regions[name] = image[y1:y2, x1:x2]
                else:
                    x1, x2 = int(w * self.regions_config[name]['x1']), int(w * self.regions_config[name]['x2'])
                    y1, y2 = int(h * self.regions_config[name]['y1']), int(h * self.regions_config[name]['y2'])
                    regions[name] = image[y1:y2, x1:x2]
        
        return regions