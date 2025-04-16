import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import os
import json
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import Dict

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
                if "text_recognition" in params["default"]:
                   text_recognition_config = params["default"]["text_recognition"]
                   self.model_type = text_recognition_config.get("model", "tesseract")
                   self.language = text_recognition_config.get("language", "en")
                   self.text_recognition_params = text_recognition_config
            except Exception as e:
                print(f"Error loading OCR or text recognition settings: {e}")
        else :
             self.model_type = 'tesseract'
             self.language = 'en'
             self.text_recognition_params = {}
        
        self.crnn_model = None
        self.donut_model = None
        self.m4c_model = None
        self.gmr_model = None
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

        if self.model_type == 'crnn':
            self.crnn_model = self._load_crnn_model()
        elif self.model_type == 'm4c':
            self.m4c_model = self._load_m4c_model()
        elif self.model_type == 'gmr':
            self.gmr_model = self._load_gmr_model()
        elif self.model_type == 'donut':
            self.donut_model = self._load_donut_model()

    def _load_crnn_model(self):
        print(f"Loading CRNN model with language: {self.language}")
        try:
            model_path = self.text_recognition_params.get("crnn", {}).get("model_path")
            if not model_path:
                raise ValueError("CRNN model path not specified in config.")
            if self.language != "en":
                language_model_path = self.text_recognition_params.get("crnn", {}).get("language_models", {}).get(self.language)
                if not language_model_path:
                    raise ValueError(f"CRNN language model path not specified for language: {self.language}")
                # Load language-specific model
                print(f"Loading CRNN language-specific model with path : {language_model_path}")
                model = ...  # Load CRNN model with language_model_path
            else:
                # Load default English model
                print(f"Loading CRNN default English model with path : {model_path}")
                model = ...  # Load CRNN model with model_path
            return model
        except Exception as e:
            print(f"Error loading CRNN model: {e}")
            return None
    
    def _load_donut_model(self):
        print(f"Loading Donut model with language: {self.language}")
        try:
            model_path = self.text_recognition_params.get("donut", {}).get("model_path")
            if not model_path:
                raise ValueError("Donut model path not specified in config.")
            if self.language != "en":
                language_model_path = self.text_recognition_params.get("donut", {}).get("language_models", {}).get(self.language)
                if not language_model_path:
                    raise ValueError(f"Donut language model path not specified for language: {self.language}")
                 # Load language-specific model
                print(f"Loading Donut language-specific model with path : {language_model_path}")
                model = ...  # Load CRNN model with language_model_path
            else:
                # Load default English model
                print(f"Loading Donut default English model with path : {model_path}")
                model = ...  # Load CRNN model with model_path
            return model
        except Exception as e:
            print(f"Error loading Donut model: {e}")
            return None
    def _load_m4c_model(self):
        print(f"Loading M4C model with language: {self.language}")
        try:
            model_path = self.text_recognition_params.get("m4c", {}).get("model_path")
            if not model_path:
                raise ValueError("M4C model path not specified in config.")
            if self.language != "en":
                language_model_path = self.text_recognition_params.get("m4c", {}).get("language_models", {}).get(self.language)
                if not language_model_path:
                    raise ValueError(f"M4C language model path not specified for language: {self.language}")
                 # Load language-specific model
                print(f"Loading M4C language-specific model with path : {language_model_path}")
                model = ...  # Load CRNN model with language_model_path
            else:
                # Load default English model
                print(f"Loading M4C default English model with path : {model_path}")
                model = ...  # Load CRNN model with model_path
            return model
        except Exception as e:
            print(f"Error loading M4C model: {e}")
            return None
    def _load_gmr_model(self):
        print(f"Loading GMR model with language: {self.language}")
        try:
            model = ...  # Load GMR model
            return model
        except Exception as e:
            print(f"Error loading GMR model: {e}")
            return None

    
    

    def _run_crnn_model(self, model, region,language):
        print("Running CRNN model...")
        return ""

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
        

    def _run_donut_model(self, model, region, language):
        if model is None:
            return ""
        try:
            print(f"Running Donut model with language: {self.language}")
            # Perform Donut inference on the region using the loaded model
            text = ...  # Perform inference and get recognized text
            return text.strip()
        except Exception as e:
            print(f"Error during Donut inference: {e}")
            return ""

    def _run_crnn_model(self, model, region, language):
        if model is None:
            return ""
        try:
            print(f"Running CRNN model with language: {self.language}")
            # Perform CRNN inference on the region using the loaded model
            text = ...  # Perform inference and get recognized text
            return text.strip()
        except Exception as e:
            print(f"Error during CRNN inference: {e}")
            return ""
    def _run_m4c_model(self, model, region, language):
        if model is None:
            return ""
        try:
            print(f"Running M4C model with language: {self.language}")
            # Perform M4C inference on the region using the loaded model
            text = ...  # Perform inference and get recognized text
            return text.strip()
        except Exception as e:
            print(f"Error during M4C inference: {e}")
            return ""
    def _run_gmr_model(self, model, region, language):
        if model is None:
            return ""
        try:
            print(f"Running GMR model with language: {self.language}")
            # Perform GMR inference on the region using the loaded model
            text = ...  # Perform inference and get recognized text
            return text.strip()
        except Exception as e:
            print(f"Error during GMR inference: {e}")
            return ""

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
        if self.model_type == 'tesseract' or force_tesseract:
            return self.recognize_text_tesseract(region, region_type)
        elif self.model_type == 'trocr' :
            return self.recognize_text_transformer(region)
        elif self.model_type == 'crnn' and self.crnn_model is not None :
             return self._run_crnn_model(self.crnn_model, region,self.language)
        elif self.model_type== 'donut':
            return self._run_donut_model(self.donut_model, region,self.language)
        elif self.model_type == 'm4c' and self.m4c_model is not None :
            return self._run_m4c_model(self.m4c_model , region,self.language)
        elif self.model_type == 'gmr' and self.gmr_model is not None:
            return self._run_gmr_model(self.gmr_model, region,self.language)
        else:
            print(f"Warning: Model type '{self.model_type}' not recognized or not loaded . Defaulting to Tesseract.")
            return self.recognize_text_tesseract(region, region_type)

    def extract_all_text(self, regions):
        text_data = {}
        for name, region in regions.items():
            if name == 'micr_line':            
                text_data[name] = self.recognize_text(region, region_type=name, force_tesseract=True) 
            else:
                text_data[name] = self.recognize_text(region, region_type=name)
        return text_data        
