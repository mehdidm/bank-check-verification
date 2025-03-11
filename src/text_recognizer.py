import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import os
import json
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TextRecognizer:
    """
    Class for performing OCR and text recognition on check regions.
    Supports multiple OCR engines and specialized configurations.
    """
    
    def __init__(self, use_transformer=False, detection_params_path=None, check_type=None):
        """
        Initialize the text recognizer.
        
        Args:
            use_transformer (bool): Whether to use transformer-based OCR.
            detection_params_path (str, optional): Path to detection parameters file.
            check_type (str, optional): Type of check to use specific parameters.
        """
        self.use_transformer = use_transformer
        self.check_type = check_type or "default"
        
        # Default OCR settings
        self.ocr_settings = {
            "contrast_ths": 0.1,
            "text_threshold": 0.7,
            "low_text": 0.4,
            "width_ths": 0.8,
            "mag_ratio": 1.5
        }
        
        # Load OCR settings from detection parameters if available
        if detection_params_path and os.path.exists(detection_params_path):
            try:
                with open(detection_params_path, 'r') as f:
                    params = json.load(f)
                    
                if self.check_type in params and "ocr_settings" in params[self.check_type]:
                    self.ocr_settings = params[self.check_type]["ocr_settings"]
                    print(f"Loaded OCR settings for check type: {self.check_type}")
            except Exception as e:
                print(f"Error loading OCR settings from detection parameters: {e}")
        
        self.tesseract_configs = {
            'micr_line': '--psm 7 -c tessedit_char_whitelist=0123456789⑆⑇ ',
            'amount_box': '--psm 7 -c tessedit_char_whitelist=0123456789,.$',
            'date_line': '--psm 7 -c tessedit_char_whitelist=0123456789/-',
            'default': '--psm 6'
        }
        
        # Initialize transformer model if requested
        if use_transformer:
            try:
                self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
                print("Transformer OCR model loaded successfully.")
            except Exception as e:
                print(f"Error loading transformer OCR model: {e}")
                print("Falling back to Tesseract OCR only.")
                self.use_transformer = False
    
    def preprocess_region(self, region):
        """
        Additional preprocessing specific to OCR.
        
        Args:
            region (numpy.ndarray): Region image.
            
        Returns:
            PIL.Image: Preprocessed image for OCR.
        """
        # Convert to PIL image format
        if isinstance(region, np.ndarray):
            # Make sure image is grayscale
            if len(region.shape) == 3:
                region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Apply additional preprocessing based on OCR settings
            # Enhance contrast
            clahe = cv2.createCLAHE(
                clipLimit=self.ocr_settings.get("contrast_ths", 0.1) * 10, 
                tileGridSize=(8, 8)
            )
            region = clahe.apply(region)
            
            # Apply adaptive thresholding if needed
            if self.ocr_settings.get("text_threshold", 0.7) > 0.5:
                binary = cv2.adaptiveThreshold(
                    region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                region = binary
            
            # Convert to PIL Image
            pil_img = Image.fromarray(region)
        else:
            pil_img = region
            
        return pil_img
    
    def recognize_text_tesseract(self, region, region_type='default'):
        """
        Recognize text using Tesseract OCR.
        
        Args:
            region (numpy.ndarray): Region image.
            region_type (str): Type of region for specialized config.
            
        Returns:
            str: Recognized text.
        """
        # Preprocess region
        pil_img = self.preprocess_region(region)
        
        # Get config for region type
        config = self.tesseract_configs.get(region_type, self.tesseract_configs['default'])
        
        # Add custom OCR settings if available
        if region_type in ['payee_line', 'written_amount'] and self.ocr_settings.get("low_text", 0.4) < 0.4:
            # Optimize for handwritten text
            config += ' --oem 3'  # LSTM only
        
        # Perform OCR
        text = pytesseract.image_to_string(pil_img, config=config)
        
        # Post-process text based on region type
        if region_type == 'micr_line':
            # Clean up MICR line (remove spaces, keep only digits and MICR symbols)
            text = re.sub(r'[^0-9⑆⑇]', '', text)
        elif region_type == 'amount_box':
            # Clean up amount (keep only digits, comma, dot, and currency symbols)
            text = re.sub(r'[^0-9,.€$]', '', text)
        elif region_type == 'date_line':
            # Clean up date (keep only digits and date separators)
            text = re.sub(r'[^0-9/-]', '', text)
        
        return text.strip()
    
    def recognize_text_transformer(self, region):
        """
        Recognize text using transformer-based OCR.
        
        Args:
            region (numpy.ndarray): Region image.
            
        Returns:
            str: Recognized text.
        """
        # Preprocess region
        pil_img = self.preprocess_region(region)
        
        # Prepare image for model
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
        
        # Generate text using OCR settings for confidence thresholds
        generated_ids = self.model.generate(
            pixel_values,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text.strip()
    
    def recognize_micr(self, region):
        """
        Specialized method for MICR line recognition.
        
        Args:
            region (numpy.ndarray): MICR line region.
            
        Returns:
            str: Recognized MICR text.
        """
        # Apply additional preprocessing for MICR
        if region is None or region.size == 0:
            return ""
        
        # Ensure region is grayscale
        if len(region.shape) == 3:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold with inverted polarity for MICR
        binary = cv2.adaptiveThreshold(
            region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Dilate to connect characters
        kernel_size = max(2, int(region.shape[1] / 300))  # Adjust kernel size based on image width
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Convert back to regular polarity for OCR
        inverted = cv2.bitwise_not(dilated)
        
        # Use specialized MICR config
        text = self.recognize_text_tesseract(inverted, region_type='micr_line')
        
        # Clean up MICR text (remove spaces, keep only digits and MICR symbols)
        text = re.sub(r'[^0-9⑆⑇]', '', text)
        
        return text
    
    def recognize_text(self, region, region_type='default', force_tesseract=False):
        """
        Recognize text using the appropriate OCR engine.
        
        Args:
            region (numpy.ndarray): Region image.
            region_type (str): Type of region for specialized config.
            force_tesseract (bool): Force using Tesseract even if transformer is available.
            
        Returns:
            str: Recognized text.
        """
        # Skip empty regions
        if region is None or region.size == 0:
            return ""
            
        # Use transformer for handwritten parts if available
        if self.use_transformer and not force_tesseract and region_type in ['signature', 'written_amount', 'payee_line']:
            return self.recognize_text_transformer(region)
        else:
            return self.recognize_text_tesseract(region, region_type)
    
    def extract_all_text(self, regions):
        """
        Extract text from all regions.
        
        Args:
            regions (dict): Dictionary of region images.
            
        Returns:
            dict: Dictionary of recognized text for each region.
        """
        text_data = {}
        
        for name, region in regions.items():
            if name == 'micr_line':
                text_data[name] = self.recognize_micr(region)
            else:
                text_data[name] = self.recognize_text(region, region_type=name)
                
        return text_data