import os
import json
import cv2
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('check_extractor')

def setup_logging(log_file=None):
    """
    Setup logging configuration.
    
    Args:
        log_file (str, optional): Path to log file.
    """
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

def load_json_config(config_path):
    """
    Load a JSON configuration file.
    
    Args:
        config_path (str): Path to the JSON config file.
        
    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}

def save_json_config(config, config_path):
    """
    Save a configuration dictionary to a JSON file.
    
    Args:
        config (dict): Configuration dictionary.
        config_path (str): Path to save the JSON config file.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved config to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")

def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """
    Get list of image files in a directory.
    
    Args:
        directory (str): Directory path.
        extensions (tuple): Tuple of valid image extensions.
        
    Returns:
        list: List of image file paths.
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return []
        
    image_files = []
    
    for file in os.listdir(directory):
        if file.lower().endswith(extensions):
            image_files.append(os.path.join(directory, file))
    
    logger.info(f"Found {len(image_files)} image files in {directory}")
    return image_files

def create_output_directory(base_dir="output", use_timestamp=True):
    """
    Create an output directory for results.
    
    Args:
        base_dir (str): Base output directory.
        use_timestamp (bool): Whether to create a timestamped subdirectory.
        
    Returns:
        str: Path to the created output directory.
    """
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, timestamp)
    else:
        output_dir = base_dir
        
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir

def resize_image(image, max_size=1600):
    """
    Resize an image while maintaining its aspect ratio.
    
    Args:
        image (numpy.ndarray): Input image.
        max_size (int): Maximum dimension (width or height).
        
    Returns:
        numpy.ndarray: Resized image.
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    if h > w:
        if h > max_size:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            return image
    else:
        if w > max_size:
            new_w = max_size
            new_h = int(h * (max_size / w))
        else:
            return image
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def normalize_text(text):
    """
    Normalize and clean up text from OCR.
    
    Args:
        text (str): Input text.
        
    Returns:
        str: Normalized text.
    """
    if not text:
        return ""
        
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common OCR errors and special characters
    text = text.replace('|', '1').replace('$', 's').replace('#', 'tt')
    
    return text.strip()

def calculate_confidence(extracted_data, required_fields=None):
    """
    Calculate a confidence score for extraction results.
    
    Args:
        extracted_data (dict): Dictionary of extracted data.
        required_fields (list, optional): List of required field names.
        
    Returns:
        float: Confidence score (0.0 to 1.0).
    """
    if required_fields is None:
        required_fields = ['amount', 'date', 'payee', 'routing_number', 'account_number']
    
    # Count non-empty required fields
    present_fields = sum(1 for field in required_fields if extracted_data.get(field))
    
    # Calculate basic confidence
    confidence = present_fields / len(required_fields)
    
    return confidence

def filter_contours_by_size(contours, min_area=100, max_area=None, image_shape=None):
    """
    Filter contours by size.
    
    Args:
        contours (list): List of contours.
        min_area (float): Minimum contour area.
        max_area (float, optional): Maximum contour area.
        image_shape (tuple, optional): Image shape (height, width).
        
    Returns:
        list: Filtered contours.
    """
    if not contours:
        return []
        
    if max_area is None and image_shape is not None:
        # Set max area to 50% of image area
        max_area = 0.5 * image_shape[0] * image_shape[1]
    elif max_area is None:
        max_area = float('inf')
    
    filtered_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    return filtered_contours