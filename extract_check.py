#!/usr/bin/env python3
"""Fast check information extraction script."""

import cv2
from check_extractor.extractor import CheckExtractor

def main():
    # Initialize extractor
    print("Initializing check extractor...")
    extractor = CheckExtractor('zitouna')
    
    # Process check image
    image_path = 'data/zitouna.jpg'
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
    if 'amount' in result:
        print("Amount:")
        print(f"  Numerical: {result['amount']['numerical']}")
        print(f"  Text: {result['amount']['text']}")
        print("-" * 40)
    
    # Print other fields
    for field, value in result.items():
        if field != 'amount':  # Skip amount as it's already printed
            print(f"{field.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()
