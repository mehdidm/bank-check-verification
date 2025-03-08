import cv2
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

class RegionDetector:
    """
    Class for detecting and extracting regions from check images.
    Supports both fixed coordinate-based extraction and dynamic detection.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the region detector.
        
        Args:
            config_path (str, optional): Path to the regions configuration file.
        """
        self.config_path = config_path
        self.regions_config = self._load_config() if config_path else self._default_config()
        
    def _load_config(self):
        """
        Load regions configuration from a JSON file.
        
        Returns:
            dict: Regions configuration.
        """
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._default_config()
    
    def _default_config(self):
        """
        Provide default regions configuration.
        
        Returns:
            dict: Default regions configuration.
        """
        return {
            'micr_line': {'y1': 0.8, 'y2': 0.95, 'x1': 0.1, 'x2': 0.9},  # bottom strip with routing/account numbers
            'amount_box': {'y1': 0.35, 'y2': 0.45, 'x1': 0.7, 'x2': 0.95},  # numeric amount
            'payee_line': {'y1': 0.25, 'y2': 0.35, 'x1': 0.15, 'x2': 0.9},  # pay to the order of line
            'date_line': {'y1': 0.15, 'y2': 0.25, 'x1': 0.7, 'x2': 0.95},  # date region
            'signature': {'y1': 0.65, 'y2': 0.8, 'x1': 0.6, 'x2': 0.95},  # signature region
            'written_amount': {'y1': 0.45, 'y2': 0.55, 'x1': 0.15, 'x2': 0.9}  # written amount
        }
    
    def extract_regions_fixed(self, image):
        """
        Extract check regions using fixed coordinates from config.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            dict: Dictionary of extracted region images.
        """
        h, w = image.shape if len(image.shape) == 2 else image.shape[:2]
        regions = {}
        
        for name, coords in self.regions_config.items():
            y_start, y_end = int(h * coords['y1']), int(h * coords['y2'])
            x_start, x_end = int(w * coords['x1']), int(w * coords['x2'])
            
            # Extract region
            regions[name] = image[y_start:y_end, x_start:x_end]
        
        return regions
    
    def detect_micr_line(self, image):
        """
        Detect MICR line using image processing techniques.
        
        Args:
            image (numpy.ndarray): Input grayscale image.
            
        Returns:
            numpy.ndarray: Detected MICR line region.
        """
        # Focus on bottom third of the image
        h, w = image.shape
        bottom_third = image[int(h*0.7):, :]
        
        # Apply horizontal morphology to detect horizontal lines
        kernel = np.ones((1, int(w/50)), np.uint8)
        morph = cv2.morphologyEx(bottom_third, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (likely the MICR line area)
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Return the MICR region from original image
            y_offset = int(image.shape[0] * 0.7)
            return image[y_offset + y:y_offset + y + h, x:x + w]
        
        # Fallback to fixed region if detection fails
        return self.extract_regions_fixed(image)['micr_line']
    
    def detect_signature(self, image):
        """
        Attempt to detect signature using contour analysis.
        
        Args:
            image (numpy.ndarray): Input grayscale image.
            
        Returns:
            numpy.ndarray: Detected signature region or None if not found.
        """
        # Focus on bottom right quadrant where signatures usually are
        h, w = image.shape
        bottom_right = image[int(h*0.5):, int(w*0.5):]
        
        # Threshold to isolate signature
        _, thresh = cv2.threshold(bottom_right, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphology to connect signature strokes
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by area to find signature-like contours
            sig_contours = [c for c in contours if cv2.contourArea(c) > (h*w*0.005)]
            
            if sig_contours:
                # Create a mask for all signature contours
                mask = np.zeros_like(bottom_right)
                cv2.drawContours(mask, sig_contours, -1, 255, -1)
                
                # Find bounding rectangle for all contours
                x_coords = []
                y_coords = []
                for c in sig_contours:
                    x, y, w, h = cv2.boundingRect(c)
                    x_coords.extend([x, x+w])
                    y_coords.extend([y, y+h])
                
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                
                # Add padding
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(bottom_right.shape[1], x2 + padding)
                y2 = min(bottom_right.shape[0], y2 + padding)
                
                # Extract signature region
                signature = bottom_right[y1:y2, x1:x2]
                
                # Map back to original image coordinates
                orig_x1 = int(w*0.5) + x1
                orig_y1 = int(h*0.5) + y1
                orig_x2 = int(w*0.5) + x2
                orig_y2 = int(h*0.5) + y2
                
                return image[orig_y1:orig_y2, orig_x1:orig_x2]
        
        # Fallback to fixed region if detection fails
        return self.extract_regions_fixed(image)['signature']
    
    def detect_regions_dynamic(self, image):
        """
        Attempt to dynamically detect regions based on image features.
        Falls back to fixed coordinates for regions that can't be detected.
        
        Args:
            image (numpy.ndarray): Input grayscale image.
            
        Returns:
            dict: Dictionary of extracted region images.
        """
        # Start with fixed regions as fallback
        regions = self.extract_regions_fixed(image)
        
        # Try to detect MICR line
        regions['micr_line'] = self.detect_micr_line(image)
        
        # Try to detect signature
        regions['signature'] = self.detect_signature(image)
        
        return regions
    
    def extract_regions(self, image, method='fixed'):
        """
        Extract regions from a check image.
        
        Args:
            image (numpy.ndarray): Input image.
            method (str): Method for extraction ('fixed' or 'dynamic').
            
        Returns:
            dict: Dictionary of extracted region images.
        """
        if method == 'dynamic':
            return self.detect_regions_dynamic(image)
        else:
            return self.extract_regions_fixed(image)
    
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