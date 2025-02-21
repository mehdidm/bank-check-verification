#!/bin/bash

# 1. Preprocess
python scripts/process.py --input_dir data/brut --output_dir data/processed_checks


# 2. Upload
python scripts/upload_to_labelstudio.py

# 3. Active learning
python scripts/active_learning.py

# Step 4: Retrain model (run nightly)
python retrain_model.py
#//fast api