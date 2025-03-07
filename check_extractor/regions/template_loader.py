"""Template loader for check region configurations."""

from check_extractor.regions import template_zitouna

def get_template(template_name):
    """Load region configuration for the specified template.
    
    Args:
        template_name (str): Name of the template to load
        
    Returns:
        dict: Region configuration dictionary
        
    Raises:
        ValueError: If template not found
    """
    templates = {
        'zitouna': template_zitouna.get_check_regions()
    }
    
    if template_name not in templates:
        raise ValueError(f"Template '{template_name}' not found")
        
    return templates[template_name]
