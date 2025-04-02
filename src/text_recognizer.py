import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import os
import json
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TextRecognizer:
    def __init__(self, use_transformer=False, detection_params_path=None, check_type=None):
        self.use_transformer = use_transformer
        self.check_type = check_type or "default"
        self.ocr_settings = {
            "contrast_ths": 0.2,
            "text_threshold": 0.6,
            "low_text": 0.3,
            "width_ths": 0.7,
            "mag_ratio": 2.0
        }
        if detection_params_path and os.path.exists(detection_params_path):
            try:
                with open(detection_params_path, 'r') as f:
                    params = json.load(f)
                if self.check_type in params and "ocr_settings" in params[self.check_type]:
                    self.ocr_settings = params[self.check_type]["ocr_settings"]
            except Exception as e:
                print(f"Error loading OCR settings: {e}")
        self.tesseract_configs = {
            'micr_line': '--psm 7 -c tessedit_char_whitelist=0123456789⑆⑇ ',
            'amount_box': '--psm 7 -c tessedit_char_whitelist=0123456789,.$',
            'date_line': '--psm 7 -c tessedit_char_whitelist=0123456789/-',
            'default': '--psm 6'
        }
        if use_transformer:
            try:
                self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
                self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
                print("Transformer OCR model loaded successfully.")
            except Exception as e:
                print(f"Error loading transformer OCR model: {e}")
                self.use_transformer = False

    def preprocess_region(self, region):
        if isinstance(region, np.ndarray):
            if len(region.shape) == 3:
                region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=self.ocr_settings.get("contrast_ths", 0.2) * 10, tileGridSize=(8, 8))
            region = clahe.apply(region)
            if self.ocr_settings.get("text_threshold", 0.6) > 0.5:
                region = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            pil_img = Image.fromarray(region)
        else:
            pil_img = region
        return pil_img

    def recognize_text_tesseract(self, region, region_type='default'):
        pil_img = self.preprocess_region(region)
        config = self.tesseract_configs.get(region_type, self.tesseract_configs['default'])
        if region_type in ['payee_line', 'written_amount']:
            config += ' --oem 3'
        text = pytesseract.image_to_string(pil_img, config=config)
        if region_type == 'micr_line':
            text = re.sub(r'[^0-9⑆⑇]', '', text)
        elif region_type == 'amount_box':
            text = re.sub(r'[^0-9,.€$]', '', text)
        elif region_type == 'date_line':
            text = re.sub(r'[^0-9/-]', '', text)
        return text.strip()

    def recognize_text_transformer(self, region):
        pil_img = self.preprocess_region(region)
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values, max_length=64, num_beams=4, early_stopping=True)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()

    def recognize_micr(self, region):
        if region is None or region.size == 0:
            return ""
        if len(region.shape) == 3:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        inverted = cv2.bitwise_not(dilated)
        text = self.recognize_text_tesseract(inverted, region_type='micr_line')
        return re.sub(r'[^0-9⑆⑇]', '', text)

    def recognize_text(self, region, region_type='default', force_tesseract=False):
        if region is None or region.size == 0:
            return ""
        if self.use_transformer and not force_tesseract and region_type in ['signature', 'written_amount', 'payee_line']:
            return self.recognize_text_transformer(region)
        return self.recognize_text_tesseract(region, region_type)

    def extract_all_text(self, regions):
        text_data = {}
        for name, region in regions.items():
            if name == 'micr_line':
                text_data[name] = self.recognize_micr(region)
            else:
                text_data[name] = self.recognize_text(region, region_type=name)
        return text_data