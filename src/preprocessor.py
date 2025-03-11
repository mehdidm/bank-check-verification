import cv2
import numpy as np
import os
import json

class ImagePreprocessor:
    """
    Class for preprocessing check images to improve OCR accuracy.
    Includes methods for enhancing image quality, removing noise,
    and improving contrast for better text recognition.
    """
    
    def __init__(self, detection_params_path=None, check_type=None):
        """
        Initialize the image preprocessor.
        
        Args:
            detection_params_path (str, optional): Path to detection parameters file.
            check_type (str, optional): Type of check to use specific parameters.
        """
        self.check_type = check_type or "default"
        
        # Default preprocessing settings
        self.preprocessing_settings = {
            "adaptive_threshold": True,
            "clahe": True,
            "denoise": True
        }
        
        # Load preprocessing settings from detection parameters if available
        if detection_params_path and os.path.exists(detection_params_path):
            try:
                with open(detection_params_path, 'r') as f:
                    params = json.load(f)
                    
                if self.check_type in params and "preprocessing" in params[self.check_type]:
                    self.preprocessing_settings = params[self.check_type]["preprocessing"]
                    print(f"Loaded preprocessing settings for check type: {self.check_type}")
            except Exception as e:
                print(f"Error loading preprocessing settings from detection parameters: {e}")
        
    def load_image(self, image_path):
        """
        Load an image from file.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            numpy.ndarray: The loaded image.
        """
        return cv2.imread(image_path)
    
    def to_grayscale(self, image):
        """
        Convert image to grayscale.
        
        Args:
            image (numpy.ndarray): Input color image.
            
        Returns:
            numpy.ndarray: Grayscale image.
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def apply_threshold(self, image, method='adaptive'):
        """
        Apply thresholding to enhance text visibility.
        
        Args:
            image (numpy.ndarray): Grayscale image.
            method (str): Thresholding method ('adaptive', 'otsu', or 'binary').
            
        Returns:
            numpy.ndarray: Thresholded image.
        """
        if method == 'adaptive':
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == 'otsu':
            _, thresh = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return thresh
        else:  # binary
            _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return thresh
    
    def denoise(self, image, strength=10):
        """
        Apply noise reduction to the image.
        
        Args:
            image (numpy.ndarray): Input image.
            strength (int): Denoising strength.
            
        Returns:
            numpy.ndarray: Denoised image.
        """
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    
    def deskew(self, image):
        """
        Deskew the image to straighten text.
        
        Args:
            image (numpy.ndarray): Input image.
            
        Returns:
            numpy.ndarray: Deskewed image.
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate moments
        moments = cv2.moments(image)
        
        # Calculate skew angle
        if moments['mu02'] > 0:
            skew = moments['mu11'] / moments['mu02']
            angle = np.degrees(np.arctan(skew))
            
            # Get rotation matrix
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            return cv2.warpAffine(
                image, M, (width, height), 
                flags=cv2.INTER_CUBIC, 
                borderMode=cv2.BORDER_REPLICATE
            )
        
        return image
    
    def enhance_contrast(self, image):
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image (numpy.ndarray): Grayscale image.
            
        Returns:
            numpy.ndarray: Contrast-enhanced image.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def preprocess(self, image_path_or_array, deskew=True, denoise_strength=10, 
                  threshold_method='adaptive', enhance=True):
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image_path_or_array (str or numpy.ndarray): Path to the image file or image array.
            deskew (bool): Whether to apply deskewing.
            denoise_strength (int): Strength of denoising.
            threshold_method (str): Method for thresholding.
            enhance (bool): Whether to enhance contrast.
            
        Returns:
            tuple: Original image and preprocessed image.
        """
        # Load the image if path is provided
        if isinstance(image_path_or_array, str):
            original = self.load_image(image_path_or_array)
        else:
            original = image_path_or_array.copy()
        
        # Convert to grayscale
        gray = self.to_grayscale(original)
        
        # Apply preprocessing based on settings
        processed = gray.copy()
        
        # Enhance contrast if requested or in settings
        if enhance and self.preprocessing_settings.get('clahe', True):
            processed = self.enhance_contrast(processed)
        
        # Apply thresholding if in settings
        if self.preprocessing_settings.get('adaptive_threshold', True):
            processed = self.apply_threshold(processed, method=threshold_method)
        
        # Denoise the image if requested or in settings
        if denoise_strength > 0 and self.preprocessing_settings.get('denoise', True):
            processed = self.denoise(processed, strength=denoise_strength)
        
        # Deskew if requested
        if deskew:
            processed = self.deskew(processed)
            
        return original, processed