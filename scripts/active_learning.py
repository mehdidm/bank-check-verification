import requests
import easyocr
from typing import List, Dict

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "941619bd68d969f5e7f9b7853d8eb918acc86876"
HEADERS = {"Authorization": f"Token {API_KEY}"}

# Initialize OCR
reader = easyocr.Reader(['fr', 'ar'])


def validate_amount(num: str, text: str, lang: str) -> bool:
    # Add your custom logic to convert words to numbers (FR/AR)
    # Example: For French, "deux cents" → 200
    return True  # Replace with actual validation


def get_uncertain_tasks(threshold=0.8) -> List[Dict]:
    # Fetch unlabeled tasks
    tasks = requests.get(f"{LABEL_STUDIO_URL}/api/tasks?project=1", headers=HEADERS).json()

    uncertain_tasks = []
    for task in tasks:
        # Run OCR prediction
        image_path = task['data']['image']
        results = reader.readtext(image_path)

        # Extract numeric and literal amounts (simplified example)
        num_amount = ...  # Parse OCR results for numeric amount
        text_amount = ...  # Parse OCR results for literal amount

        # Validate
        if not validate_amount(num_amount, text_amount, lang='fr'):
            uncertain_tasks.append(task)

    return uncertain_tasks[:100]  # Limit to top 100 uncertain tasks


if __name__ == "__main__":
    uncertain_tasks = get_uncertain_tasks()
    # Push to Label Studio's "Review" queue
    for task in uncertain_tasks:
        requests.post(f"{LABEL_STUDIO_URL}/api/tasks/{task['id']}/flag", headers=HEADERS)