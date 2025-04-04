import re
import json
import os
import cv2
import numpy as np
import pytesseract
from typing import Dict, Optional, Tuple

class DataExtractor:
    def __init__(self, patterns_config_path=None):
        self.patterns_config_path = patterns_config_path
        self.patterns = self._load_patterns() if patterns_config_path else self._default_patterns()

        # Configure tesseract path - adjust this based on your installation
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Define check regions (normalized coordinates)
        self.regions = {
            'date': (0.75, 0.05, 0.95, 0.15),      # (x1, y1, x2, y2) as percentages
            'payee': (0.15, 0.20, 0.95, 0.30),
            'amount_box': (0.75, 0.20, 0.95, 0.30),
            'written_amount': (0.15, 0.30, 0.95, 0.40),
            'signature': (0.65, 0.55, 0.95, 0.75),
            'micr': (0.10, 0.90, 0.90, 0.98)
        }

    def _load_patterns(self):
        try:
            with open(self.patterns_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading patterns config: {e}")
            return self._default_patterns()

    def _default_patterns(self):
        return {
            'routing_number': r'\b([0-9]{9})\b',
            'account_number': r'\b([0-9]{10,14})\b',
            'check_number': r'\b([0-9]{3,6})\b',
            'date': r'\b(0[1-9]|1[0-2])[-/\.](0[1-9]|[12][0-9]|3[01])[-/\.]((19|20)\d{2}|\d{2})\b',
            'amount': r'\$?\s*([0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2}|[0-9]+\.[0-9]{2})',
            'written_amount': r'(?:(?:pay to the order of|pay).{0,50})?((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|and|-|\s|dollars|cents)+)'
        }

    def extract_routing_number(self, micr_text):
        pattern = self.patterns['routing_number']
        match = re.search(pattern, micr_text)
        return match.group(1) if match else ""

    def extract_account_number(self, micr_text):
        pattern = self.patterns['account_number']
        match = re.search(pattern, micr_text)
        return match.group(1) if match else ""

    def extract_check_number(self, micr_text):
        pattern = self.patterns['check_number']
        match = re.search(pattern, micr_text)
        return match.group(1) if match else ""

    def extract_date(self, date_text):
        pattern = self.patterns['date']
        match = re.search(pattern, date_text)
        if match:
            month, day, year = match.groups()[0], match.groups()[1], match.groups()[2]
            year = f"20{year}" if len(year) == 2 and int(year) <= 99 else year
            return f"{month}/{day}/{year}"
        return ""

    def extract_amount(self, amount_text):
        clean_text = amount_text.replace(' ', '')
        pattern = self.patterns['amount']
        match = re.search(pattern, clean_text)
        return match.group(1).replace(',', '') if match else ""

    def extract_payee(self, payee_text):
        payee_pattern = r'(?:pay to the order of|pay to|payto)[ :]*([^\n\r$]+)'
        match = re.search(payee_pattern, payee_text.lower())
        if match:
            payee = match.group(1).strip()
            payee = re.sub(r'(?:dollars|and \d+/100|\d+/100).*$', '', payee, flags=re.IGNORECASE).strip()
            return payee
        return payee_text.strip()

    def extract_written_amount(self, written_amount_text):
        pattern = self.patterns['written_amount']
        match = re.search(pattern, written_amount_text.lower())
        if match:
            written_amount = match.group(1).strip()
            written_amount = re.sub(r'(dollars|only)$', '', written_amount, flags=re.IGNORECASE).strip()
            written_amount = re.sub(r'\s+', ' ', written_amount)
            return written_amount
        return ""

    def process_check_image(self, image_path: str) -> Dict[str, str]:
        """Process a check image and extract all relevant information."""
        # Read and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract data from each region
        results = {}
        results['date'] = self._extract_from_region(gray, 'date')
        results['payee'] = self._extract_from_region(gray, 'payee')
        results['amount'] = self._extract_from_region(gray, 'amount_box')
        results['written_amount'] = self._extract_from_region(gray, 'written_amount')
        results['micr_data'] = self._process_micr(gray)
        
        # Process extracted data using existing patterns
        results['date'] = self.extract_date(results['date'])
        results['payee'] = self.extract_payee(results['payee'])
        results['amount'] = self.extract_amount(results['amount'])
        results['written_amount'] = self.extract_written_amount(results['written_amount'])
        
        # Extract MICR components
        micr_text = results['micr_data']
        results['routing_number'] = self.extract_routing_number(micr_text)
        results['account_number'] = self.extract_account_number(micr_text)
        results['check_number'] = self.extract_check_number(micr_text)
        
        return results

    def _extract_from_region(self, image: np.ndarray, region_name: str) -> str:
        """Extract text from a specific region of the check."""
        h, w = image.shape
        x1, y1, x2, y2 = self.regions[region_name]
        
        # Convert normalized coordinates to pixel coordinates
        x1, y1 = int(w * x1), int(h * y1)
        x2, y2 = int(w * x2), int(h * y2)
        
        # Extract region
        roi = image[y1:y2, x1:x2]
        
        # Apply region-specific preprocessing
        if region_name == 'micr':
            # MICR-specific preprocessing
            roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        else:
            # General preprocessing for other regions
            roi = cv2.GaussianBlur(roi, (3, 3), 0)
            roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Perform OCR
        config = '--psm 6'  # Assume uniform block of text
        if region_name == 'amount_box':
            config = '--psm 7 -c tessedit_char_whitelist=0123456789,.'
        
        text = pytesseract.image_to_string(roi, config=config).strip()
        return text

    def _process_micr(self, image: np.ndarray) -> str:
        """Special processing for MICR line."""
        h, w = image.shape
        x1, y1, x2, y2 = self.regions['micr']
        
        # Convert normalized coordinates to pixel coordinates
        x1, y1 = int(w * x1), int(h * y1)
        x2, y2 = int(w * x2), int(h * y2)
        
        # Extract MICR region
        micr_roi = image[y1:y2, x1:x2]
        
        # MICR-specific preprocessing
        micr_roi = cv2.threshold(micr_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Use tesseract with MICR configuration
        config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        micr_text = pytesseract.image_to_string(micr_roi, config=config).strip()
        
        return micr_text