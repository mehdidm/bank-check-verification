"""Utility functions for check information extraction."""

import cv2
import numpy as np
import easyocr
from PIL import Image
import os

# Initialize EasyOCR reader with English language support
_reader = easyocr.Reader(['en'], gpu=False)

def clean_text(text):
    """Clean extracted text by removing common OCR errors."""
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s.,/]', '', text)
    text = ' '.join(text.split())
    
    # Fix common OCR errors
    text = text.replace('l', 'I').replace('O', 'o')
    text = text.replace('0', 'O').replace('1', 'I')
    
    # Fix common address abbreviations
    text = text.replace('ST.', 'STREET')
    text = text.replace('AVE.', 'AVENUE')
    text = text.replace('RD.', 'ROAD')
    text = text.replace('BLVD.', 'BOULEVARD')
    text = text.replace('DR.', 'DRIVE')
    
    # Fix common province abbreviations
    text = text.replace('ON', 'ONTARIO')
    text = text.replace('BC', 'BRITISH COLUMBIA')
    text = text.replace('AB', 'ALBERTA')
    text = text.replace('QC', 'QUEBEC')
    text = text.replace('MB', 'MANITOBA')
    text = text.replace('SK', 'SASKATCHEWAN')
    text = text.replace('NS', 'NOVA SCOTIA')
    text = text.replace('NB', 'NEW BRUNSWICK')
    text = text.replace('PE', 'PRINCE EDWARD ISLAND')
    text = text.replace('NL', 'NEWFOUNDLAND AND LABRADOR')
    text = text.replace('NT', 'NORTHWEST TERRITORIES')
    text = text.replace('NU', 'NUNAVUT')
    text = text.replace('YT', 'YUKON')
    
    return text.strip()

def preprocess_image(image):
    """Preprocess image for better OCR results."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize if too small
    min_width = 800
    if gray.shape[1] < min_width:
        scale = min_width / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Save debug image
    cv2.imwrite('debug_preprocessed.png', gray)
    
    return gray

def extract_text_from_image(image, region_config):
    """Extract text from a region of the image using EasyOCR."""
    # Extract region coordinates
    x = int(region_config['x'] * image.shape[1] / 100)
    y = int(region_config['y'] * image.shape[0] / 100)
    w = int(region_config['width'] * image.shape[1] / 100)
    h = int(region_config['height'] * image.shape[0] / 100)
    
    # Add padding to region
    padding = 2
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2*padding)
    h = min(image.shape[0] - y, h + 2*padding)
    
    # Extract region
    roi = image[y:y+h, x:x+w]
    
    # Save ROI for inspection
    cv2.imwrite(f'debug_roi_{region_config.get("is_payee_name", "unknown")}.png', roi)
    
    try:
        # Use EasyOCR with default settings
        result = _reader.readtext(roi)
        if result:
            # Extract text from result tuples
            text = ' '.join([item[1] for item in result])
            print(f"EasyOCR result for {region_config.get('is_payee_name', 'unknown')}: {text}")
            return text.strip()
        return ''
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ''
