"""Region configuration management for check templates."""

import os
import importlib
import json
from typing import Dict, Optional

class TemplateLoader:
    """Load and manage check template configurations."""
    
    def __init__(self):
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all available template configurations."""
        template_dir = os.path.dirname(__file__)
        for filename in os.listdir(template_dir):
            if filename.startswith('template_') and filename.endswith('.py'):
                template_name = filename[9:-3]  # Remove 'template_' prefix and '.py' extension
                try:
                    module = importlib.import_module(f'.{filename[:-3]}', package='check_extractor.regions')
                    self.templates[template_name] = module.REGIONS
                except (ImportError, AttributeError) as e:
                    print(f"Warning: Failed to load template {template_name}: {e}")
    
    def get_template(self, name: str) -> Optional[Dict]:
        """Get a specific template configuration by name."""
        return self.templates.get(name)
    
    def list_templates(self) -> list:
        """List all available template names."""
        return list(self.templates.keys())
    
    def get_region(self, template_name: str, region_name: str) -> Optional[Dict]:
        """Get a specific region configuration from a template."""
        template = self.get_template(template_name)
        if template:
            return template.get(region_name)
        return None
    
    def get_ocr_config(self, template_name: str, region_name: str) -> Dict:
        """Get OCR configuration for a specific region."""
        region = self.get_region(template_name, region_name)
        if region and 'ocr_config' in region:
            return region['ocr_config']
        # Return default OCR config optimized for check processing
        return {
            'contrast_ths': 0.2,
            'text_threshold': 0.6,
            'low_text': 0.3,
            'width_ths': 0.7,
            'mag_ratio': 2.5
        }
