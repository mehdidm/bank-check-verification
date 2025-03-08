"""Utility functions for check information extraction."""

import cv2
import numpy as np
import easyocr

# Initialize EasyOCR reader with French language support
_reader = easyocr.Reader(['fr'], gpu=False)

def preprocess_image(image):
    """Preprocess image for better OCR results.
    
    Args:
        image: OpenCV image in BGR format
        
    Returns:
        preprocessed: Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize if too small
    min_width = 2000
    if gray.shape[1] < min_width:
        scale = min_width / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,  # Larger block size for better contrast
        5    # Higher C value for better contrast
    )
    
    return binary

def extract_text_from_image(image, region_config):
    """Extract text from a region of the image using EasyOCR.
    
    Args:
        image: Preprocessed image
        region_config: Region configuration dictionary with coordinates and OCR settings
        
    Returns:
        str: Extracted text
    """
    # Extract region coordinates
    x = int(region_config['x'] * image.shape[1] / 100)
    y = int(region_config['y'] * image.shape[0] / 100)
    w = int(region_config['width'] * image.shape[1] / 100)
    h = int(region_config['height'] * image.shape[0] / 100)
    
    # Extract region
    roi = image[y:y+h, x:x+w]
    
    # Get OCR settings optimized for handwriting
    ocr_settings = {
        'text_threshold': 0.5,  # Lower threshold for handwriting
        'low_text': 0.3,       # Better small text detection
        'width_ths': 0.5,      # More flexible width threshold
        'mag_ratio': 2.5,      # Better small text detection
        'contrast_ths': 0.3,   # Better handling of varying contrast
        'add_margin': 0.1,     # Add margin around text
        'beamWidth': 5         # Increased accuracy
    }
    
    # Override with region-specific settings
    if 'text_threshold' in region_config:
        ocr_settings['text_threshold'] = region_config['text_threshold']
    if 'width_ths' in region_config:
        ocr_settings['width_ths'] = region_config['width_ths']
    if 'mag_ratio' in region_config:
        ocr_settings['mag_ratio'] = region_config['mag_ratio']
    
    # Perform OCR
    try:
        result = _reader.readtext(
            roi,
            detail=0,
            paragraph=False,
            **ocr_settings
        )
        text = ' '.join(result) if result else ''
        
        # Clean up text based on field type
        if region_config.get('is_check_number'):
            text = ''.join(filter(str.isdigit, text))
        elif region_config.get('is_amount_numerical'):
            text = ''.join(c for c in text if c.isdigit() or c in '.,')
        elif region_config.get('is_amount_text'):
            text = ' '.join(text.split())
            # Remove any digits from amount text
            text = ''.join(c for c in text if not c.isdigit() or c.isspace())
        elif region_config.get('is_date'):
            text = ''.join(c for c in text if c.isdigit() or c in '/-.')
            parts = [p for p in text.replace('/', '-').split('-') if p]
            if len(parts) == 3:
                text = '/'.join(parts)
        elif region_config.get('is_account_number'):
            text = ''.join(filter(str.isalnum, text))
        elif region_config.get('is_payee_name'):
            text = ' '.join(text.split())
            # Remove any digits from payee name
            text = ''.join(c for c in text if not c.isdigit() or c.isspace())
        
        return text.strip()
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ''
