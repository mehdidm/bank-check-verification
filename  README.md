# Bank Check Processing System

## Overview
An advanced system for automated bank check processing, featuring:
- Image preprocessing
- Optical Character Recognition (OCR)
- Check validation
- Annotation support

## Features
- Multi-language OCR support
- Advanced image preprocessing
- Intelligent text extraction
- Comprehensive check validation
- Flexible annotation workflow

## Installation

### Prerequisites
- Python 3.8+
- Tesseract OCR
- Docker (optional)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/bank-check-processing.git
cd bank-check-processing

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

## Usage

### Preprocessing and OCR
```python
from src.preprocessing import CheckPreprocessor
from src.ocr_extraction import CheckProcessor

# Initialize processors
preprocessor = CheckPreprocessor()
ocr_processor = CheckProcessor()

# Preprocess check image
preprocessed_image = preprocessor.preprocess_image('check.jpg')

# Extract text
easyocr_results = ocr_processor.extract_text_easyocr(preprocessed_image)
tesseract_results = ocr_processor.extract_text_tesseract(preprocessed_image)
```

### Validation
```python
from src.check_validator import CheckValidator

validator = CheckValidator()
is_valid, message = validator.validate_amount(
    numeric_amount, 
    letter_amount
)
```

### Annotation
```python
from src.annotation_utils import LabelStudioAnnotator

annotator = LabelStudioAnnotator()
project = annotator.create_project(
    'sample_check.jpg', 
    'preprocessed_check.jpg'
)
```

## Project Structure
```
bank-check-processing/
│
├── data/
│   ├── raw_checks/
│   ├── preprocessed/
│   └── annotated/
│
├── notebooks/
│   └── check_processing_demo.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── ocr_extraction.py
│   ├── check_validator.py
│   └── annotation_utils.py
│
├── label_studio_config/
│   └── config.json
│
├── requirements.txt
└── main.py
```

## Supported Languages
- French
- English
- Arabic (Configurable)

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License
```

This comprehensive project provides a robust, flexible system for bank check processing with:
1. Advanced preprocessing
2. Multi-engine OCR
3. Intelligent validation
4. Annotation support

Would you like me to elaborate on any specific aspect of the project?