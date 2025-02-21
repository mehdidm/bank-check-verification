import os
from datetime import time

import requests
import json

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "941619bd68d969f5e7f9b7853d8eb918acc86876"
HEADERS = {"Authorization": f"Token {API_KEY}"}


def upload_batch(image_dir, batch_size=100):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i + batch_size]
        tasks = []
        for filename in batch:
            tasks.append({
                "data": {
                    "image": f"/home/mehdi/PycharmProjects/PythonProject/Ai_project/data/brut/{filename}"
                }
            })

        # Upload tasks
        upload_response = requests.post(
            f"{LABEL_STUDIO_URL}/api/projects/1/import",
            headers=HEADERS,
            json=tasks
        )
        print(f"Uploaded batch {i // batch_size + 1}: Status {upload_response.status_code}")

        # Request predictions for auto-annotation (wait 5 sec for processing)
        time.sleep(5)
        predict_response = requests.post(
            f"{LABEL_STUDIO_URL}/api/projects/1/predict",
            headers=HEADERS
        )
        print(f"Auto-annotated batch {i // batch_size + 1}: Status {predict_response.status_code}")


if __name__ == "__main__":
    upload_batch("processed_checks/", batch_size=100)