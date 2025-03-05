import easyocr
import pytesseract
import cv2
import re
import logging
from unidecode import unidecode


class CheckProcessor:
    """
    Advanced OCR processing for bank checks
    Supports multiple OCR engines and text extraction strategies
    """

    def __init__(self, languages=['fr', 'en']):
        """
        Initialize multiple OCR engines

        Args:
            languages (list): List of languages to support
        """
        self.logger = logging.getLogger(__name__)

        try:
            self.easyocr_reader = easyocr.Reader(languages)
            self.tesseract_languages = '+'.join(languages)
        except Exception as e:
            self.logger.error(f"OCR initialization error: {e}")
            raise

    def extract_text_easyocr(self, preprocessed_image):
        """
        Extract text using EasyOCR with advanced parsing

        Args:
            preprocessed_image (numpy.ndarray): Preprocessed check image

        Returns:
            dict: Categorized extracted text
        """
        try:
            results = self.easyocr_reader.readtext(preprocessed_image)

            extracted_texts = {
                'amounts': [],
                'bank_details': [],
                'other_text': []
            }

            for detection in results:
                text = unidecode(detection[1]).strip()

                # Categorize text
                if self._is_amount(text):
                    extracted_texts['amounts'].append(text)
                elif self._is_bank_detail(text):
                    extracted_texts['bank_details'].append(text)
                else:
                    extracted_texts['other_text'].append(text)

            return extracted_texts

        except Exception as e:
            self.logger.error(f"EasyOCR extraction error: {e}")
            return {'amounts': [], 'bank_details': [], 'other_text': []}

    def extract_text_tesseract(self, preprocessed_image):
        """
        Extract text using Tesseract OCR

        Args:
            preprocessed_image (numpy.ndarray): Preprocessed check image

        Returns:
            dict: Extracted and parsed text
        """
        try:
            config = f'--oem 3 --psm 6 -l {self.tesseract_languages}'
            text = pytesseract.image_to_string(preprocessed_image, config=config)

            return self._parse_tesseract_text(text)

        except Exception as e:
            self.logger.error(f"Tesseract extraction error: {e}")
            return {'full_text': '', 'amounts': [], 'bank_details': []}

    def _is_amount(self, text):
        """
        Identify if text represents a monetary amount

        Args:
            text (str): Text to analyze

        Returns:
            bool: True if text represents an amount
        """
        amount_pattern = r'^[\d\s,.]+$'
        return re.match(amount_pattern, text) is not None

    def _is_bank_detail(self, text):
        """
        Identify potential bank details

        Args:
            text (str): Text to analyze

        Returns:
            bool: True if text might be bank-related
        """
        bank_keywords = [
            'bank', 'compte', 'account', 'banque',
            'iban', 'bic', 'credit', 'cheque'
        ]
        return any(keyword in text.lower() for keyword in bank_keywords)

    def _parse_tesseract_text(self, text):
        """
        Parse and clean Tesseract OCR output

        Args:
            text (str): Raw OCR text

        Returns:
            dict: Parsed text components
        """
        # Remove extra whitespaces
        cleaned_text = ' '.join(text.split())

        # Extract potential amounts and bank details
        amounts = re.findall(r'\b\d+[.,]?\d*\b', cleaned_text)
        bank_details = [
            word for word in cleaned_text.split()
            if self._is_bank_detail(word)
        ]

        return {
            'full_text': cleaned_text,
            'amounts': amounts,
            'bank_details': bank_details
        }