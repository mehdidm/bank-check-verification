import os
import logging
from src.preprocessing import CheckPreprocessor
from src.ocr_extraction import CheckProcessor
from src.annotation_utils import LabelStudioAnnotator
from src.check_validator import CheckValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='check_processing.log'
)


def main():
    """
    Main execution script for bank check processing system
    """
    try:
        # Initialize processors
        preprocessor = CheckPreprocessor(debug=True)
        ocr_processor = CheckProcessor()
        annotator = LabelStudioAnnotator()
        validator = CheckValidator()

        # Directories
        raw_checks_dir = 'data/raw_checks'
        preprocessed_dir = 'data/preprocessed'

        # Create necessary directories
        os.makedirs(preprocessed_dir, exist_ok=True)

        # Process each check image
        for filename in os.listdir(raw_checks_dir):
            if filename.lower().endswith(('.jpg', '.png', '.tiff', '.pdf')):
                # Full path to the image
                image_path = os.path.join(raw_checks_dir, filename)

                try:
                    # Preprocess the image
                    preprocessed_image = preprocessor.preprocess_image(image_path)

                    # Save preprocessed image
                    preprocessed_path = os.path.join(preprocessed_dir, f'preprocessed_{filename}')
                    preprocessor.save_image(preprocessed_image, preprocessed_path)

                    # Extract text using multiple OCR methods
                    easyocr_results = ocr_processor.extract_text_easyocr(preprocessed_image)
                    tesseract_results = ocr_processor.extract_text_tesseract(preprocessed_image)

                    # Validate check details
                    if easyocr_results['amounts'] and tesseract_results['amounts']:
                        is_valid, validation_message = validator.validate_amount(
                            easyocr_results['amounts'][0],
                            tesseract_results['amounts'][0]
                        )

                    # Create annotation project
                    annotator.create_project(filename, preprocessed_path)

                    # Log processing results
                    logging.info(f"Processed {filename}: Validation - {is_valid}")

                except Exception as file_error:
                    logging.error(f"Error processing {filename}: {file_error}")

    except Exception as main_error:
        logging.critical(f"Critical error in main execution: {main_error}")


if __name__ == "__main__":
    main()