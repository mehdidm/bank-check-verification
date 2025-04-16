import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import os
import json
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import Dict

from transformers import DonutProcessor, VisionEncoderDecoderForCausalLM

class TextRecognizer:

    def __init__(self, use_transformer=False, detection_params_path=None, check_type=None):
        
        
        self.use_transformer = use_transformer
        self.check_type = check_type or "default"
        self.ocr_settings = {
            "contrast_ths": 0.2,
            "new_contrast":10,
            "tileGridSize":8,
            "ADAPTIVE_THRESH_GAUSSIAN_C":11,
            "ADAPTIVE_THRESH": 2,

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
        else:
            self.model_type = 'tesseract'
            self.language = 'en'
            self.text_recognition_params = {}

        self.trocr_processor = None
        self.trocr_model = None        self.crnn_model = None
        self.donut_model = None
        self.donut_processor = None
        self.m4c_model = None
        self.gmr_model = None
        self.tesseract_configs = {
            "contrast_ths": 0.2,
            "new_contrast":10,
            "tileGridSize":8,
            "ADAPTIVE_THRESH_GAUSSIAN_C":11,
            "ADAPTIVE_THRESH":2,

            'micr_line': '--psm 7 -c tessedit_char_whitelist=0123456789⑆⑇ ',
            'amount_box': '--psm 7 -c tessedit_char_whitelist=0123456789,.$',

            'date_line': '--psm 7 -c tessedit_char_whitelist=0123456789/-',
            'default': '--psm 6'
        }
        if use_transformer:
            
            try:
                self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to("cuda")
                print("Transformer OCR model loaded successfully.")
            except Exception as e:
                print(f"Error loading transformer OCR model: {e}")
                self.use_transformer = False
        try:
            self.donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
            self.donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
            print("Donut model loaded successfully.").to("cuda")
        except Exception as e:
            print(f"Error loading Donut model: {e}")
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
                # Load default English model.
                print(f"Loading CRNN default English model with path : {model_path}")
                model = ...  # Load CRNN model with model_path
            return model
        except Exception as e:
            print(f"Error loading CRNN model: {e}")
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
                    raise ValueError(f"M4C language model path not specified for language:{self.language}")
                 # Load language-specific model
                print(f"Loading M4C language-specific model with path : {language_model_path}")
                model = ...  # Load M4C model with language_model_path
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
            model = ...  # Load GMR model.
            return model
        except Exception as e:
            print(f"Error loading GMR model: {e}")
            return None

    def _run_crnn_model(self, model, region,language):
        print(f"Running CRNN model with language: {language}")
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
        try:
            if self.trocr_processor is None or self.trocr_model is None:
                print("TrOCR model or processor not loaded.")
                return ""
            pixel_values = self.trocr_processor(pil_img, return_tensors="pt").pixel_values
            generated_ids = self.trocr_model.generate(pixel_values.to(self.trocr_model.device), max_length=64, num_beams=4, early_stopping=True)
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
        except Exception as e:
            print(f"Error during TrOCR inference: {e}")
            return ""
    def recognize_text_donut(self, region):
        try:
            if self.donut_model is None or self.donut_processor is None:
                print("Donut model or processor not loaded.")
                return ""
            try:
                pil_img = self.preprocess_region(region)
                pixel_values = self.donut_processor(pil_img, return_tensors="pt").pixel_values.to(self.donut_model.device)

                task_prompt = "<s_cord-v2>"
                decoder_input_ids = self.donut_processor.tokenizer(
                    task_prompt,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).input_ids.to(self.donut_model.device)

                outputs = self.donut_model.generate(pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512,
                pad_token_id=self.donut_processor.tokenizer.pad_token_id,
                eos_token_id=self.donut_processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.donut_processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                decoder_input_ids=decoder_input_ids.to(self.donut_model.device),
                max_length=self.donut_model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.donut_processor.tokenizer.pad_token_id,
                eos_token_id=self.donut_processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=5,
                bad_words_ids=[[self.donut_processor.tokenizer.unk_token_id]], #not allow unknow token
                return_dict_in_generate=True,
                )
                text = self.donut_processor.batch_decode(outputs.sequences)[0].replace(self.donut_processor.tokenizer.eos_token, "").replace(self.donut_processor.tokenizer.pad_token, "")
                return text.strip()
        except Exception as e:
            print(f"Error during Donut inference: {e}")
            return ""

        # 2. Check for Agreement
        if trocr_output == donut_output:
            print("TrOCR and Donut outputs are identical.")
            return trocr_output  # Or donut_output, as they are the same

        # 3. Heuristics and Type-Specific Checks
        if region_type in ['amount_box', 'micr_line', 'date_line']:
            # Prioritize TrOCR for numeric/symbolic fields
            print(f"Prioritizing TrOCR for region type: {region_type}")
            return trocr_output
        elif region_type in ['payee_line', 'written_amount']:
            # Prioritize Donut for more complex, potentially multi-line text
            print(f"Prioritizing Donut for region type: {region_type}")
            return donut_output
        else:
            # Default: Prefer TrOCR (you could change this preference)
            print("Using default preference: TrOCR")
            return trocr_output        

    def _run_crnn_model(self, model, region, language):
        if model is None:
            return ""
        try:
            print(f"Running CRNN model with language : {language}")
            # Perform CRNN inference on the region using the loaded model
            text = ...  # Perform inference and get recognized text
            return text.strip()
        except Exception as e:
            print(f"Error during CRNN inference: {e}")
            return ""
    def _compare_and_combine_ocr(self, trocr_output, donut_output, region_type):
        print(f"Comparing TrOCR and Donut outputs for region type : {region_type}")
        if trocr_output == donut_output:
            return trocr_output
        else:        
            if len(trocr_output)> len(donut_output):
                return trocr_output
            else:
            print(f"Running CRNN model with language : {self.language}")
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
            print(f"Running M4C model with language : {self.language}")
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
            print(f"Running GMR model with language : {self.language}")
            # Perform GMR inference on the region using the loaded model
            text = ...  # Perform inference and get recognized text
            return text.strip()
        except Exception as e:
            print(f"Error during GMR inference: {e}")
            return ""
    
    def recognize_micr(self, region):
        if region is None or region.size == 0:
         return ""
        if len(region.shape)== 3 :
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
        elif self.model_type == 'trocr':
            trocr_output = self.recognize_text_transformer(region)
            donut_output = self.recognize_text_donut(region)
            return self._compare_and_combine_ocr(trocr_output, donut_output, region_type)
        elif self.model_type == 'crnn' and self.crnn_model is not None:
            return self._run_crnn_model(self.crnn_model, region, self.language)
        elif self.model_type == 'donut':
            trocr_output = self.recognize_text_transformer(region)
            donut_output = self.recognize_text_donut(region)
            return self._compare_and_combine_ocr(trocr_output, donut_output, region_type)
        elif self.model_type == 'm4c' and self.m4c_model is not None:
            return self._run_m4c_model(self.m4c_model, region, self.language)
        elif self.model_type == 'gmr' and self.gmr_model is not None:
            return self._run_gmr_model(self.gmr_model, region, self.language)
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