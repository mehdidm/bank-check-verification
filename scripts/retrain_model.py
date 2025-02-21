import requests
import json
from collections import defaultdict

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "941619bd68d969f5e7f9b7853d8eb918acc86876"
HEADERS = {"Authorization": f"Token {API_KEY}"}

def export_annotations():
    response = requests.get(
        f"{LABEL_STUDIO_URL}/api/projects/1/export?format=JSON",
        headers=HEADERS
    )
    return response.json()

def prepare_training_data(annotations):
    training_data = defaultdict(list)
    for ann in annotations:
        image_path = ann['data']['image']
        for region in ann['annotations'][0]['result']:
            if region['type'] == 'rectanglelabels':
                label = region['value']['rectanglelabels'][0]
                text = ...  # Extract transcribed text from adjacent TextArea
                training_data[label].append((image_path, text))
    return training_data

# Example: Retrain EasyOCR (custom training requires model modification)
# See EasyOCR docs for advanced retraining: https://github.com/JaidedAI/EasyOCR

if __name__ == "__main__":
    annotations = export_annotations()
    training_data = prepare_training_data(annotations)
    # Add retraining logic here