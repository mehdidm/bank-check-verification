import cv2
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

class RegionDetector:
    """
    Class for detecting and extracting regions from check images.
    """
    
    def __init__(self, config_path=None, detection_params_path=None, check_type=None):
        """
        Initialize the region detector.
        
        Args:
            config_path (str, optional): Path to the regions configuration file.
            detection_params_path (str, optional): Path to detection parameters file.
            check_type (str, optional): Type of check to use specific parameters.
        """
        self.config_path = config_path
        self.regions_config = {}
        self.check_type = check_type or "default"
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.regions_config = json.load(f)
            except Exception as e:
                print(f"Error loading regions configuration: {e}")
        
        # Load detection parameters
        self.detection_params = self._load_detection_params(detection_params_path)
    
    def _load_detection_params(self, detection_params_path=None):
        """
        Load detection parameters from file or use defaults.
        
        Args:
            detection_params_path (str, optional): Path to detection parameters file.
            
        Returns:
            dict: Detection parameters.
        """
        default_params = {
            "micr_line": {
                "focus_area": {"y1": 0.7, "y2": 1.0, "x1": 0.0, "x2": 1.0},
                "kernel_width_factor": 50,
                "min_contour_area": 100
            },
            "amount_box": {
                "focus_area": {"y1": 0.0, "y2": 0.4, "x1": 0.6, "x2": 1.0},
                "min_area": 100,
                "aspect_ratio_min": 2.0,
                "aspect_ratio_max": 5.0
            },
            "payee_line": {
                "focus_area": {"y1": 0.25, "y2": 0.5, "x1": 0.0, "x2": 0.8},
                "kernel_width_factor": 30,
                "min_line_length_factor": 0.33,
                "max_line_gap": 20
            },
            "date_line": {
                "focus_area": {"y1": 0.0, "y2": 0.2, "x1": 0.6, "x2": 1.0},
                "kernel_width": 15,
                "min_line_length_factor": 0.1,
                "max_line_gap": 10
            },
            "signature": {
                "focus_area": {"y1": 0.5, "y2": 0.8, "x1": 0.5, "x2": 1.0},
                "padding": 10
            },
            "written_amount": {
                "focus_area": {"y1": 0.35, "y2": 0.55, "x1": 0.1, "x2": 0.8},
                "kernel_width_factor": 30,
                "min_line_length_factor": 0.25,
                "max_line_gap": 20
            }
        }
        
        if detection_params_path and os.path.exists(detection_params_path):
            try:
                with open(detection_params_path, 'r') as f:
                    params = json.load(f)
                    
                # Use specified check type or default
                if self.check_type in params:
                    return params[self.check_type]
                else:
                    print(f"Check type '{self.check_type}' not found in parameters, using default")
                    return params.get("default", default_params)
            except Exception as e:
                print(f"Error loading detection parameters: {e}")
        
        return default_params
    
    def extract_regions(self, image, method='dynamic'):
        """
        Extract regions from the check image using the specified method.
        
        Args:
            image (numpy.ndarray): The check image.
            method (str): The region extraction method ('fixed', 'dynamic').
            
        Returns:
            dict: A dictionary of extracted regions.
        """
        if method == 'fixed' and self.regions_config:
            return self.extract_regions_fixed(image)
        else:
            return self.extract_regions_dynamic(image)
    
    def extract_regions_fixed(self, image):
        """
        Extract regions from the check image using fixed coordinates.
        
        Args:
            image (numpy.ndarray): The check image.
            
        Returns:
            dict: A dictionary of extracted regions.
        """
        h, w = image.shape if len(image.shape) == 2 else image.shape[:2]
        regions = {}
        
        for name, coords in self.regions_config.items():
            y_start, y_end = int(h * coords['y1']), int(h * coords['y2'])
            x_start, x_end = int(w * coords['x1']), int(w * coords['x2'])
            regions[name] = image[y_start:y_end, x_start:x_end]
        
        return regions
    
    def extract_regions_dynamic(self, image):
        """
        Extract regions from the check image using dynamic detection.
        
        Args:
            image (numpy.ndarray): The check image.
            
        Returns:
            dict: A dictionary of extracted regions.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Get image dimensions
        h, w = gray.shape
        
        # Initialize regions dictionary
        regions = {}
        
        # Detect MICR line (bottom part of check)
        regions['micr_line'] = self.detect_micr_line(gray)
        
        # Detect amount box (usually top-right corner)
        regions['amount_box'] = self.detect_amount_box(gray)
        
        # Detect payee line (usually middle part after "Pay to the order of")
        regions['payee_line'] = self.detect_payee_line(gray)
        
        # Detect date line (usually top-right)
        regions['date_line'] = self.detect_date_line(gray)
        
        # Detect signature area (usually bottom-right)
        regions['signature'] = self.detect_signature(gray)
        
        # Detect written amount (usually middle part)
        regions['written_amount'] = self.detect_written_amount(gray)
        
        return regions
    
    def detect_micr_line(self, image):
        """
        Detect the MICR line in the check image.
        
        Args:
            image (numpy.ndarray): The grayscale check image.
            
        Returns:
            numpy.ndarray: The MICR line region.
        """
        h, w = image.shape
        
        # Get parameters for MICR line detection
        params = self.detection_params.get("micr_line", {})
        focus_area = params.get("focus_area", {"y1": 0.7, "y2": 1.0, "x1": 0.0, "x2": 1.0})
        kernel_width_factor = params.get("kernel_width_factor", 50)
        min_contour_area = params.get("min_contour_area", 100)
        
        # Focus on the bottom part of the image where MICR line is typically located
        y_start = int(h * focus_area["y1"])
        y_end = int(h * focus_area["y2"])
        x_start = int(w * focus_area["x1"])
        x_end = int(w * focus_area["x2"])
        
        bottom_part = image[y_start:y_end, x_start:x_end]
        
        # Apply morphological operations to enhance horizontal lines
        kernel = np.ones((1, int(w/kernel_width_factor)), np.uint8)
        morph = cv2.morphologyEx(bottom_part, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        
        if valid_contours:
            # Find the largest contour which is likely the MICR line
            largest = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Return the MICR line region with original coordinates
            return bottom_part[y:y + h, x:x + w]
        
        # Fallback to fixed region if detection fails
        return self._get_default_region(image, 'micr_line')
    
    def detect_amount_box(self, image):
        """
        Detect the amount box in the check image.
        
        Args:
            image (numpy.ndarray): The grayscale check image.
            
        Returns:
            numpy.ndarray: The amount box region.
        """
        h, w = image.shape
        
        # Get parameters for amount box detection
        params = self.detection_params.get("amount_box", {})
        focus_area = params.get("focus_area", {"y1": 0.0, "y2": 0.4, "x1": 0.6, "x2": 1.0})
        min_area = params.get("min_area", 100)
        aspect_ratio_min = params.get("aspect_ratio_min", 2.0)
        aspect_ratio_max = params.get("aspect_ratio_max", 5.0)
        
        # Focus on the top-right part where amount box is typically located
        y_start = int(h * focus_area["y1"])
        y_end = int(h * focus_area["y2"])
        x_start = int(w * focus_area["x1"])
        x_end = int(w * focus_area["x2"])
        
        top_right = image[y_start:y_end, x_start:x_end]
        
        # Apply adaptive thresholding to enhance box edges
        thresh = cv2.adaptiveThreshold(top_right, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations to enhance box structure
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio to find rectangular boxes
        amount_box = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:  # Skip small contours
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:  # Avoid division by zero
                continue
                
            aspect_ratio = w / float(h)
            
            # Amount boxes typically have aspect ratio between specified min and max
            if aspect_ratio_min <= aspect_ratio <= aspect_ratio_max and area > max_area:
                amount_box = (x, y, w, h)
                max_area = area
        
        if amount_box:
            x, y, w, h = amount_box
            
            # Return the amount box region
            return top_right[y:y + h, x:x + w]
        
        # Fallback to fixed region if detection fails
        return self._get_default_region(image, 'amount_box')
    
    def detect_payee_line(self, image):
        """
        Detect the payee line in the check image.
        
        Args:
            image (numpy.ndarray): The grayscale check image.
            
        Returns:
            numpy.ndarray: The payee line region.
        """
        h, w = image.shape
        
        # Get parameters for payee line detection
        params = self.detection_params.get("payee_line", {})
        focus_area = params.get("focus_area", {"y1": 0.25, "y2": 0.5, "x1": 0.0, "x2": 0.8})
        kernel_width_factor = params.get("kernel_width_factor", 30)
        min_line_length_factor = params.get("min_line_length_factor", 0.33)
        max_line_gap = params.get("max_line_gap", 20)
        
        # Focus on the middle-left part where payee line is typically located
        y_start = int(h * focus_area["y1"])
        y_end = int(h * focus_area["y2"])
        x_start = int(w * focus_area["x1"])
        x_end = int(w * focus_area["x2"])
        
        middle_left = image[y_start:y_end, x_start:x_end]
        
        # Apply horizontal line detection
        kernel = np.ones((1, int(w/kernel_width_factor)), np.uint8)
        morph = cv2.morphologyEx(middle_left, cv2.MORPH_CLOSE, kernel)
        
        # Find horizontal lines
        edges = cv2.Canny(morph, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=w*min_line_length_factor, 
                               maxLineGap=max_line_gap)
        
        if lines is not None and len(lines) > 0:
            # Find the longest horizontal line which is likely the payee line
            max_length = 0
            payee_line = None
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10:  # Ensure it's a horizontal line
                    length = abs(x2 - x1)
                    if length > max_length:
                        max_length = length
                        payee_line = (min(x1, x2), y1, max(x1, x2), y2)
            
            if payee_line:
                x1, y1, x2, y2 = payee_line
                
                # Add some padding
                padding = 5
                y_min = max(0, y1 - padding)
                y_max = min(middle_left.shape[0], y2 + padding)
                
                # Return the payee line region
                return middle_left[y_min:y_max, x1:x2]
        
        # Fallback to fixed region if detection fails
        return self._get_default_region(image, 'payee_line')
    
    def detect_date_line(self, image):
        """
        Detect the date line in the check image.
        
        Args:
            image (numpy.ndarray): The grayscale check image.
            
        Returns:
            numpy.ndarray: The date line region.
        """
        h, w = image.shape
        
        # Get parameters for date line detection
        params = self.detection_params.get("date_line", {})
        focus_area = params.get("focus_area", {"y1": 0.0, "y2": 0.2, "x1": 0.6, "x2": 1.0})
        kernel_width = params.get("kernel_width", 15)
        min_line_length_factor = params.get("min_line_length_factor", 0.1)
        max_line_gap = params.get("max_line_gap", 10)
        
        # Focus on the top-right corner where date line is typically located
        y_start = int(h * focus_area["y1"])
        y_end = int(h * focus_area["y2"])
        x_start = int(w * focus_area["x1"])
        x_end = int(w * focus_area["x2"])
        
        top_right = image[y_start:y_end, x_start:x_end]
        
        # Apply horizontal line detection
        kernel = np.ones((1, kernel_width), np.uint8)
        morph = cv2.morphologyEx(top_right, cv2.MORPH_CLOSE, kernel)
        
        # Find horizontal lines
        edges = cv2.Canny(morph, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=w*min_line_length_factor, 
                               maxLineGap=max_line_gap)
        
        if lines is not None and len(lines) > 0:
            # Sort lines by y-coordinate (top to bottom)
            sorted_lines = sorted(lines, key=lambda line: line[0][1])
            
            # The first line is likely the date line
            x1, y1, x2, y2 = sorted_lines[0][0]
            
            # Add some padding
            padding = 5
            y_min = max(0, y1 - padding)
            y_max = min(top_right.shape[0], y2 + padding)
            
            # Return the date line region
            return top_right[y_min:y_max, min(x1, x2):max(x1, x2)]
        
        # Fallback to fixed region if detection fails
        return self._get_default_region(image, 'date_line')
    
    def detect_signature(self, image):
        """
        Detect the signature area in the check image.
        
        Args:
            image (numpy.ndarray): The grayscale check image.
            
        Returns:
            numpy.ndarray: The signature area region.
        """
        h, w = image.shape
        
        # Get parameters for signature detection
        params = self.detection_params.get("signature", {})
        focus_area = params.get("focus_area", {"y1": 0.5, "y2": 0.8, "x1": 0.5, "x2": 1.0})
        padding = params.get("padding", 10)
        
        # Focus on the bottom-right part where signature is typically located
        y_start = int(h * focus_area["y1"])
        y_end = int(h * focus_area["y2"])
        x_start = int(w * focus_area["x1"])
        x_end = int(w * focus_area["x2"])
        
        bottom_right = image[y_start:y_end, x_start:x_end]
        
        # Apply adaptive thresholding to enhance signature
        thresh = cv2.adaptiveThreshold(bottom_right, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Combine all contours to get the signature area
            all_contours = np.vstack([contour for contour in contours])
            x, y, w, h = cv2.boundingRect(all_contours)
            
            # Add padding
            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(bottom_right.shape[1], x + w + padding)
            y_max = min(bottom_right.shape[0], y + h + padding)
            
            # Return the signature area
            return bottom_right[y_min:y_max, x_min:x_max]
        
        # Fallback to fixed region if detection fails
        return self._get_default_region(image, 'signature')
    
    def detect_written_amount(self, image):
        """
        Detect the written amount area in the check image.
        
        Args:
            image (numpy.ndarray): The grayscale check image.
            
        Returns:
            numpy.ndarray: The written amount region.
        """
        h, w = image.shape
        
        # Get parameters for written amount detection
        params = self.detection_params.get("written_amount", {})
        focus_area = params.get("focus_area", {"y1": 0.35, "y2": 0.55, "x1": 0.1, "x2": 0.8})
        kernel_width_factor = params.get("kernel_width_factor", 30)
        min_line_length_factor = params.get("min_line_length_factor", 0.25)
        max_line_gap = params.get("max_line_gap", 20)
        
        # Focus on the middle part where written amount is typically located
        y_start = int(h * focus_area["y1"])
        y_end = int(h * focus_area["y2"])
        x_start = int(w * focus_area["x1"])
        x_end = int(w * focus_area["x2"])
        
        middle = image[y_start:y_end, x_start:x_end]
        
        # Apply horizontal line detection
        kernel = np.ones((1, int(w/kernel_width_factor)), np.uint8)
        morph = cv2.morphologyEx(middle, cv2.MORPH_CLOSE, kernel)
        
        # Find horizontal lines
        edges = cv2.Canny(morph, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=w*min_line_length_factor, 
                               maxLineGap=max_line_gap)
        
        if lines is not None and len(lines) > 0:
            # Sort lines by y-coordinate (top to bottom)
            sorted_lines = sorted(lines, key=lambda line: line[0][1])
            
            if len(sorted_lines) >= 2:
                # The written amount is typically between two horizontal lines
                x1_top, y1_top, x2_top, y2_top = sorted_lines[0][0]
                x1_bottom, y1_bottom, x2_bottom, y2_bottom = sorted_lines[1][0]
                
                # Return the written amount region
                return middle[y1_top:y1_bottom, 0:middle.shape[1]]
        
        # Fallback to fixed region if detection fails
        return self._get_default_region(image, 'written_amount')
    
    def _get_default_region(self, image, region_name):
        """
        Get a default region based on typical check layout.
        
        Args:
            image (numpy.ndarray): The check image.
            region_name (str): The name of the region.
            
        Returns:
            numpy.ndarray: The default region.
        """
        h, w = image.shape if len(image.shape) == 2 else image.shape[:2]
        
        # Default regions based on typical check layout
        default_regions = {
            'micr_line': {'y1': 0.85, 'y2': 0.95, 'x1': 0.05, 'x2': 0.95},
            'amount_box': {'y1': 0.15, 'y2': 0.25, 'x1': 0.8, 'x2': 0.95},
            'payee_line': {'y1': 0.3, 'y2': 0.4, 'x1': 0.2, 'x2': 0.8},
            'date_line': {'y1': 0.05, 'y2': 0.15, 'x1': 0.7, 'x2': 0.95},
            'signature': {'y1': 0.6, 'y2': 0.75, 'x1': 0.6, 'x2': 0.95},
            'written_amount': {'y1': 0.4, 'y2': 0.5, 'x1': 0.2, 'x2': 0.8}
        }
        
        # Use default region if available, otherwise return the center of the image
        if region_name in default_regions:
            coords = default_regions[region_name]
            y_start, y_end = int(h * coords['y1']), int(h * coords['y2'])
            x_start, x_end = int(w * coords['x1']), int(w * coords['x2'])
            return image[y_start:y_end, x_start:x_end]
        else:
            # Return center of the image as fallback
            center_h, center_w = h // 2, w // 2
            return image[center_h - 50:center_h + 50, center_w - 100:center_w + 100]
    
    def visualize_regions(self, original_image, regions):
        """
        Visualize the extracted regions.
        
        Args:
            original_image (numpy.ndarray): Original image.
            regions (dict): Dictionary of extracted region images.
            
        Returns:
            None
        """
        # Convert OpenCV BGR to RGB for matplotlib
        if len(original_image.shape) == 3:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Create a figure
        fig = plt.figure(figsize=(15, 10))
        
        # Plot original image
        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(original_rgb)
        ax.set_title("Original Check")
        ax.axis('off')
        
        # Plot regions
        region_titles = {
            'micr_line': 'MICR Line',
            'amount_box': 'Amount',
            'payee_line': 'Payee',
            'date_line': 'Date',
            'signature': 'Signature',
            'written_amount': 'Written Amount'
        }
        
        i = 2
        for name, region in regions.items():
            if i <= 8 and name in region_titles:
                ax = fig.add_subplot(2, 4, i)
                if len(region.shape) == 3:
                    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                    ax.imshow(region_rgb)
                else:
                    ax.imshow(region, cmap='gray')
                ax.set_title(region_titles.get(name, name))
                ax.axis('off')
                i += 1
        
        plt.tight_layout()
        plt.show()
        
    def save_regions_to_files(self, regions, output_dir):
        """
        Save extracted regions to files.
        
        Args:
            regions (dict): Dictionary of extracted region images.
            output_dir (str): Output directory for saved regions.
            
        Returns:
            dict: Dictionary mapping region names to file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_paths = {}
        
        for name, region in regions.items():
            file_path = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(file_path, region)
            file_paths[name] = file_path
            
        return file_paths