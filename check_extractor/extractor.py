"""Check information extractor."""

from check_extractor.regions import template_loader
from check_extractor.utils import extract_text_from_image, preprocess_image

class CheckExtractor:
    """Extract information from check images."""
    
    def __init__(self, template_name='zitouna'):
        """Initialize check extractor.
        
        Args:
            template_name (str): Name of the template to use
        """
        self.template_name = template_name
        self.template = template_loader.get_template(template_name)
        
    def extract_all_fields(self, image):
        """Extract all fields from check image.
        
        Args:
            image: OpenCV image in BGR format
            
        Returns:
            dict: Extracted field values
        """
        # Preprocess image once for all fields
        preprocessed = preprocess_image(image)
        
        # Extract all fields
        results = {}
        for field, config in self.template.items():
            text = extract_text_from_image(preprocessed, config)
            results[field] = text
            
        return results
    
    def extract_field(self, image, field_name):
        """Extract a specific field from the check image."""
        if field_name not in self.template:
            raise ValueError(f"Field '{field_name}' not found in template '{self.template_name}'")
        
        # Preprocess and extract
        preprocessed = preprocess_image(image)
        return extract_text_from_image(preprocessed, self.template[field_name])
    
    def available_fields(self):
        """Get list of available fields in the current template."""
        return list(self.template.keys())
