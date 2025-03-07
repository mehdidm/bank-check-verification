"""Check information extractor."""

import cv2
import json
from check_extractor.regions import template_loader
from check_extractor.utils import extract_text_from_image, preprocess_image
import easyocr

# Initialize EasyOCR with optimized settings
_reader = easyocr.Reader(
    ['en'],
    gpu=False,
    model_storage_directory='./models',
    download_enabled=True,
    quantize=True
)

class CheckExtractor:
    """Extract information from check images."""
    
    def __init__(self, template='canadian'):
        """Initialize the extractor.
        
        Args:
            template: Template name to use for region configuration
        """
        self.template = template
        self.regions_config = self._load_regions_config()
    
    def _load_regions_config(self):
        """Load region configuration from JSON file."""
        with open('data/regions_config_template1.json', 'r') as f:
            return json.load(f)
    
    def extract_all_fields(self, image):
        """Extract all fields from the check image.
        
        Args:
            image: OpenCV image in BGR format
            
        Returns:
            dict: Dictionary containing extracted fields
        """
        # Preprocess image
        processed = preprocess_image(image)
        
        # Extract text from each region
        result = {}
        for field, config in self.regions_config.items():
            text = extract_text_from_image(processed, config)
            if text:
                result[field] = text
        
        return result
    
    def extract_field(self, image, field_name):
        """Extract a specific field from the check image."""
        if field_name not in self.regions_config:
            raise ValueError(f"Field '{field_name}' not found in configuration")
        
        # Preprocess and extract
        processed = preprocess_image(image)
        return extract_text_from_image(processed, self.regions_config[field_name])
    
    def available_fields(self):
        """Get list of available fields in the current configuration."""
        return list(self.regions_config.keys())
