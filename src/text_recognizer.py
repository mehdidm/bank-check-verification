import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TextRecognizer:
    """
    Class for performing OCR and text recognition on check regions.
    Supports multiple OCR engines and specialized configurations.
    """
    
    def __init__(self, use_transformer=False):
        """
        Initialize the text recognizer.
        
        Args:
            use_transformer (bool): Whether to use transformer-based OCR.
        """
        self.use_transformer = use_transformer
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
        
        # Perform OCR
        text = pytesseract.image_to_string(pil_img, config=config)
        
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
        
        # Generate text
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text.strip()
    
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
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Dilate to connect characters
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Convert back to regular polarity for OCR
        inverted = cv2.bitwise_not(dilated)
        
        # Use specialized MICR config
        text = self.recognize_text_tesseract(inverted, region_type='micr_line')
        
        # Clean up text (remove non-numeric characters except special MICR symbols)
        text = re.sub(r'[^0-9⑆⑇]', '', text)
        
        return text.strip()
    
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