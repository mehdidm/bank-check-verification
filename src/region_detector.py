import cv2
import numpy as np
import json
import os
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import pytesseract
from PIL import Image
import tensorflow as tf

class RegionDetector:
    def __init__(self, config_path=None, detection_params_path=None, check_type=None):
        self.config_path = config_path
        self.detection_params_path = detection_params_path
        self.check_type = check_type or "default"
        self.regions_config = self._load_regions_config()
        self.detection_params = self._load_detection_params()
        
        # Initialize OCR models
        self.text_detector = self._initialize_text_detector()
        self.handwriting_recognizer = self._initialize_handwriting_recognizer()

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

    def _initialize_text_detector(self):
        # Configure Tesseract parameters for better accuracy
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,./$'
        return custom_config

    def _initialize_handwriting_recognizer(self):
        # Load pre-trained handwriting recognition model
        # This is a placeholder - you would need to implement actual model loading
        try:
            model = tf.keras.models.load_model('path_to_handwriting_model')
            return model
        except:
            print("Warning: Handwriting recognition model not found. Falling back to basic OCR.")
            return None

    def _detect_contours(self, image):
        # Enhanced contour detection with preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Edge detection with optimized parameters
        edges = cv2.Canny(denoised, 30, 150)
        
        # Dilate edges to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours based on area and aspect ratio
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small contours
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.2 <= aspect_ratio <= 5:  # Filter out extreme aspect ratios
                filtered_contours.append(contour)
        
        return filtered_contours

    def _classify_regions(self, contours, image_shape):
        h, w = image_shape
        features = []
        
        # Extract enhanced features for each contour
        for contour in contours:
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            hull = ConvexHull(contour.reshape(-1, 2))
            hull_area = hull.area
            
            # Calculate relative positions and normalized metrics
            center_x = (x + w_contour/2) / w
            center_y = (y + h_contour/2) / h
            normalized_area = area / (w * h)
            compactness = (perimeter ** 2) / area if area > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0
            
            features.append([
                center_x, center_y,  # Position
                w_contour/w, h_contour/h,  # Size
                normalized_area,  # Area
                compactness, solidity  # Shape features
            ])
        
        if not features:
            return {}
        
        # Use DBSCAN for more robust clustering
        features_array = np.array(features)
        db = DBSCAN(eps=0.3, min_samples=2).fit(features_array)
        labels = db.labels_
        
        # Initialize regions dictionary with confidence scores
        regions = {}
        confidences = {}
        
        # Assign regions based on position and characteristics
        for i, (contour, label, feature) in enumerate(zip(contours, labels, features)):
            if label == -1:  # Skip noise
                continue
                
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            center_x, center_y = feature[0], feature[1]
            
            # Define region assignment rules with confidence scores
            region_scores = {
                'micr_line': self._calculate_micr_confidence(center_y, feature),
                'amount_box': self._calculate_amount_confidence(center_x, center_y, feature),
                'payee_line': self._calculate_payee_confidence(center_x, center_y, feature),
                'date_line': self._calculate_date_confidence(center_x, center_y, feature),
                'signature': self._calculate_signature_confidence(center_x, center_y, feature)
            }
            
            # Assign region to highest confidence match
            best_region = max(region_scores.items(), key=lambda x: x[1])
            if best_region[1] > 0.5:  # Confidence threshold
                regions[best_region[0]] = (x, y, x + w_contour, y + h_contour)
                confidences[best_region[0]] = best_region[1]
        
        return regions, confidences

    def _calculate_micr_confidence(self, center_y, feature):
        # MICR line is typically at the bottom with specific characteristics
        bottom_position = center_y > 0.8
        appropriate_size = 0.05 < feature[3] < 0.15  # Height
        return 0.8 if (bottom_position and appropriate_size) else 0.2

    def _calculate_amount_confidence(self, center_x, center_y, feature):
        # Amount box is typically in the top-right
        position_score = 1.0 if (center_x > 0.7 and center_y < 0.3) else 0.2
        size_score = 1.0 if (feature[2] < 0.3 and feature[3] < 0.2) else 0.2
        return (position_score + size_score) / 2

    def _calculate_payee_confidence(self, center_x, center_y, feature):
        # Payee line is typically in the middle-left
        position_score = 1.0 if (center_x < 0.6 and 0.2 < center_y < 0.4) else 0.2
        size_score = 1.0 if (feature[2] > 0.3) else 0.2
        return (position_score + size_score) / 2

    def _calculate_date_confidence(self, center_x, center_y, feature):
        # Date is typically in the top-right
        position_score = 1.0 if (center_x > 0.7 and center_y < 0.2) else 0.2
        size_score = 1.0 if (feature[2] < 0.2 and feature[3] < 0.1) else 0.2
        return (position_score + size_score) / 2

    def _calculate_signature_confidence(self, center_x, center_y, feature):
        # Signature is typically in the bottom-right
        position_score = 1.0 if (center_x > 0.6 and 0.6 < center_y < 0.8) else 0.2
        size_score = 1.0 if (0.2 < feature[2] < 0.4 and 0.1 < feature[3] < 0.2) else 0.2
        return (position_score + size_score) / 2

    def extract_regions(self, image, method='dynamic'):
        h, w = image.shape[:2]
        regions = {}
        
        if method == 'dynamic':
            contours = self._detect_contours(image)
            detected_regions, confidences = self._classify_regions(contours, (h, w))
            
            # Process each region with appropriate OCR method
            for name, coords in detected_regions.items():
                x1, y1, x2, y2 = coords
                region_image = image[y1:y2, x1:x2]
                
                if name == 'micr_line':
                    # Use specialized MICR font recognition
                    text = self._process_micr(region_image)
                elif name in ['amount_box', 'date_line']:
                    # Use combined OCR for printed and handwritten numbers
                    text = self._process_numeric(region_image)
                elif name == 'signature':
                    # Use signature verification
                    text = self._verify_signature(region_image)
                else:
                    # Use general handwriting recognition
                    text = self._process_handwriting(region_image)
                
                regions[name] = {
                    'image': region_image,
                    'text': text,
                    'confidence': confidences.get(name, 0.0)
                }
        else:
            # Fallback to fixed regions if dynamic detection fails
            regions = self._extract_fixed_regions(image)
        
        return regions

    def _process_micr(self, image):
        # Specialized MICR font recognition
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,./$'
        return pytesseract.image_to_string(image, config=custom_config)

    def _process_numeric(self, image):
        # Combined OCR for numbers
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,.$'
        return pytesseract.image_to_string(image, config=custom_config)

    def _process_handwriting(self, image):
        # Use handwriting recognition model if available
        if self.handwriting_recognizer:
            # Preprocess image for the model
            processed_image = self._preprocess_for_handwriting(image)
            # Get prediction from model
            return self._predict_handwriting(processed_image)
        else:
            # Fallback to Tesseract
            return pytesseract.image_to_string(image, config=self.text_detector)

    def _verify_signature(self, image):
        # Placeholder for signature verification
        return "Signature detected"

    def _extract_fixed_regions(self, image):
        # Implementation of _extract_fixed_regions method
        pass

    def _preprocess_for_handwriting(self, image):
        # Implementation of _preprocess_for_handwriting method
        pass

    def _predict_handwriting(self, image):
        # Implementation of _predict_handwriting method
        pass