import os
import cv2
import json
import pytesseract
import pandas as pd
from tqdm import tqdm  # Pour une barre de progression

# Configurer les chemins
# Nouveau code (correct) :
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Pointe vers /Ai_project/
DATA_BRUT = os.path.join(BASE_DIR, "data", "brut")
DATA_ANNOT = os.path.join(BASE_DIR, "data", "annotations")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "exports", "resultats.csv")


def process_image(image_path, annotation_path):
    """Traite une image et son annotation."""
    # Charger l'image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Charger l'annotation
    with open(annotation_path) as f:
        data = json.load(f)

    results = {"Fichier": os.path.basename(image_path)}
    for entry in data:
        for annotation in entry["annotations"]:
            for region in annotation["result"]:
                label = region["value"]["rectanglelabels"][0]
                # Convertir les % en pixels
                x = int((region["value"]["x"] / 100) * w)
                y = int((region["value"]["y"] / 100) * h)
                width = int((region["value"]["width"] / 100) * w)
                height = int((region["value"]["height"] / 100) * h)

                # Découper la ROI
                roi = image[y:y + height, x:x + width]

                # Prétraitement pour améliorer l'OCR
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                # OCR
                text = pytesseract.image_to_string(thresh, lang="fra").strip()
                results[label] = text

    return results


# Traiter toutes les images
all_results = []
for img_file in tqdm(os.listdir(DATA_BRUT)):
    if img_file.endswith((".jpg", ".png")):
        # Chemins des fichiers
        image_path = os.path.join(DATA_BRUT, img_file)
        base_name = os.path.splitext(img_file)[0]
        annotation_path = os.path.join(DATA_ANNOT, f"{base_name}.json")

        if os.path.exists(annotation_path):
            results = process_image(image_path, annotation_path)
            all_results.append(results)
        else:
            print(f"⚠️ Annotation manquante pour {img_file}")

# Sauvegarder en CSV
df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Résultats exportés vers {OUTPUT_CSV}")