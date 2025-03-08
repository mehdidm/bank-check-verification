import cv2
import numpy as np
import pytesseract
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from .utils import preprocess_image, extract_text_from_image

class CheckExtractor:
    def __init__(self, template_dir: str = 'data'):
        """Initialize the CheckExtractor with template configurations."""
        self.template_dir = template_dir
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, Dict]:
        """Load all template configurations from the template directory."""
        templates = {}
        template_files = Path(self.template_dir).glob('regions_config_*.json')
        
        for template_file in template_files:
            template_name = template_file.stem.replace('regions_config_', '')
            with open(template_file, 'r', encoding='utf-8') as f:
                templates[template_name] = json.load(f)
        
        return templates
    
    def detect_template(self, image: np.ndarray) -> Optional[str]:
        """Detect which template the check matches using template matching."""
        best_match = None
        best_score = -1
        
        # Convert image to grayscale for template matching
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try to match with each template
        for template_name in self.templates:
            # Extract a reference region that should be consistent across templates
            # For example, the check number region
            ref_region = self.templates[template_name]['check_number']
            x, y = int(ref_region['x'] * image.shape[1] / 100), int(ref_region['y'] * image.shape[0] / 100)
            w, h = int(ref_region['width'] * image.shape[1] / 100), int(ref_region['height'] * image.shape[0] / 100)
            
            # Extract the region
            region = gray[y:y+h, x:x+w]
            
            # Calculate a similarity score (you can adjust this method)
            # For now, we'll use a simple histogram comparison
            hist = cv2.calcHist([region], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Compare with other templates
            for other_name in self.templates:
                if other_name != template_name:
                    other_region = self.templates[other_name]['check_number']
                    ox, oy = int(other_region['x'] * image.shape[1] / 100), int(other_region['y'] * image.shape[0] / 100)
                    ow, oh = int(other_region['width'] * image.shape[1] / 100), int(other_region['height'] * image.shape[0] / 100)
                    
                    other_region = gray[oy:oy+oh, ox:ox+ow]
                    other_hist = cv2.calcHist([other_region], [0], None, [256], [0, 256])
                    other_hist = cv2.normalize(other_hist, other_hist).flatten()
                    
                    # Calculate similarity score
                    score = cv2.compareHist(hist, other_hist, cv2.HISTCMP_CORREL)
                    
                    if score > best_score:
                        best_score = score
                        best_match = template_name
        
        return best_match if best_score > 0.8 else None
    
    def extract_check_info(self, image_path: str) -> Dict:
        """Extract information from a check image."""
        # Read and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Get the first template
        if not self.templates:
            raise ValueError("No templates found")
        
        template_name = list(self.templates.keys())[0]
        template = self.templates[template_name]
        
        # Preprocess the image
        preprocessed = preprocess_image(image)
        
        # Extract information using the template configuration
        check_info = {}
        for field, region in template.items():
            text = extract_text_from_image(preprocessed, region)
            check_info[field] = text
        
        # Add template information
        check_info['template'] = template_name
        
        return check_info

    def process_directory(self, directory: str) -> list:
        """Process all check images in a directory."""
        results = []
        for image_path in Path(directory).glob('*.jpg'):
            try:
                check_info = self.extract_check_info(str(image_path))
                check_info['image_path'] = str(image_path)
                results.append(check_info)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        return results 