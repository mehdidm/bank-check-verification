#!/usr/bin/env python3
"""Fast check information extraction script."""

import cv2
import json
from check_extractor.extractor import CheckExtractor

def main():
    # Load region configuration
    with open('data/regions_config_template1.json', 'r') as f:
        regions_config = json.load(f)
    
    # Initialize extractor
    print("Initializing check extractor...")
    extractor = CheckExtractor('canadian')
    
    # Process check image
    image_path = 'data/CanadianChequeSample.png'
    print(f"\nProcessing check image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    print(f"Image loaded successfully (size: {image.shape})")
    
    # Extract fields
    print("\nExtracting check information...")
    result = extractor.extract_all_fields(image)
    
    # Print results
    print("\nExtracted Check Information:")
    print("-" * 40)
    
    # Print amount first
    if 'amount_in_numbers' in result:
        print("Amount:")
        print(f"  Numerical: {result['amount_in_numbers']}")
        print(f"  Text: {result.get('amount_in_words', '')}")
        print("-" * 40)
    
    # Print other fields
    field_order = [
        'payee_name', 'address', 'date', 'bank_name_branch',
        'bank_address', 'memo', 'security_features'
    ]
    
    for field in field_order:
        if field in result:
            print(f"{field.replace('_', ' ').title()}: {result[field]}")

if __name__ == "__main__":
    main()
