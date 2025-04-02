# utils/helpers.py
import os
import logging
import json
from datetime import datetime

def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_json_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_image_files(directory):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(exts)]

def create_output_directory(base_dir="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def calculate_confidence(extracted_data):
    required_fields = ['amount', 'date', 'payee', 'routing_number', 'account_number']
    complete_fields = sum(1 for field in required_fields if extracted_data.get(field))
    return complete_fields / len(required_fields)