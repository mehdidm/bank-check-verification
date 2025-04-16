# src/model_evaluator.py

def calculate_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def calculate_region_detection_metrics(ground_truth, predictions):
    print("Calculating region detection metrics...")
    # Expected input format:
    # ground_truth: list of dicts, where each dict represents an image and contains
    #               region annotations in the format {region_name: [x1, y1, x2, y2]}
    # predictions: list of dicts, with the same format as ground_truth, but containing
    #              predicted regions
    # Example:
    # ground_truth = [
    #     {"image1": {"micr_line": [100, 450, 600, 500], "amount_box": [400, 100, 500, 150]}},
    #     {"image2": {"micr_line": [150, 400, 650, 450], "amount_box": [450, 150, 550, 200]}},
    # ]
    # predictions = [
    #     {"image1": {"micr_line": [110, 460, 590, 490], "amount_box": [410, 110, 490, 140]}},
    #     {"image2": {"micr_line": [140, 410, 660, 440], "amount_box": [440, 140, 560, 210]}},
    # ]

    # Use a library like pycocotools for mAP calculation
    # (This requires converting the input data to the COCO format)
    # For IOU and Recall, you can use libraries like scikit-learn or implement
    # the calculations manually.

    # Placeholder implementation:
    mAP = 0.0  # Replace with actual mAP calculation
    iou_values = []
    recall_values = []
    for gt, pred in zip(ground_truth, predictions):
        for region_name in gt.keys():
            gt_region = gt.get(region_name)
            pred_region = pred.get(region_name)
            if gt_region and pred_region:
                # Calculate IOU for the region
                iou = calculate_iou(gt_region, pred_region)  # Replace with actual IOU calculation
                iou_values.append(iou)
                # Calculate Recall.
                # In a real implementation, this would require a more sophisticated approach,
                # such as using a library like scikit-learn or implementing a more detailed
                # calculation based on true/false positives and negatives.
                recall = 1.0 if iou > 0.5 else 0.0  # Replace with actual Recall calculation
                recall_values.append(recall)
    mean_iou = sum(iou_values) / len(iou_values) if iou_values else 0.0
    mean_recall = sum(recall_values) / len(recall_values) if recall_values else 0.0
    return {"mAP": mAP, "IOU": mean_iou, "Recall": mean_recall}

def calculate_cer(gt_text, pred_text):
    # Placeholder for CER calculation
    # In a real implementation, you would use a library like jiwer.
    # For example:
    # import jiwer
    # cer = jiwer.cer(gt_text, pred_text)
    if len(gt_text) == 0:
        return 0.0 if len(pred_text) == 0 else 1.0
    return float(sum(1 for a, b in zip(gt_text, pred_text) if a != b) + abs(len(gt_text) - len(pred_text))) / len(gt_text)

def calculate_wer(gt_text, pred_text):
    # Placeholder for WER calculation
    # In a real implementation, you would use a library like jiwer.
    # wer = jiwer.wer(gt_text, pred_text)
    gt_words = gt_text.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return float(sum(1 for a, b in zip(gt_words, pred_words) if a != b) + abs(len(gt_words) - len(pred_words))) / len(gt_words)

def calculate_text_recognition_metrics(ground_truth, predictions):
    print("Calculating text recognition metrics...")
    # Expected input format:
    # ground_truth: list of dicts, where each dict represents an image and contains
    #               text annotations in the format {region_name: text}
    # predictions: list of dicts, with the same format as ground_truth, but containing
    #              predicted text
    # Example:
    # ground_truth = [
    #     {"image1": {"micr_line": "123456789", "amount_box": "$100.00"}},
    #     {"image2": {"micr_line": "987654321", "amount_box": "$200.00"}},
    # ]
    # predictions = [
    #     {"image1": {"micr_line": "123456780", "amount_box": "$100.00"}},
    #     {"image2": {"micr_line": "987654321", "amount_box": "$200.00"}},
    # ]

    # Use a library like jiwer for CER and WER calculation
    # (This requires extracting the text from the input data and providing it
    #  as lists of strings to the jiwer library)

    # Placeholder implementation:
    cer_values = []
    wer_values = []
    for gt, pred in zip(ground_truth, predictions):
        for region_name in gt.keys():
            gt_text = gt.get(region_name)
            pred_text = pred.get(region_name)
            if gt_text and pred_text:
                # Calculate CER and WER for the region
                cer = calculate_cer(gt_text, pred_text)  # Replace with actual CER calculation
                wer = calculate_wer(gt_text, pred_text) 
                cer_values.append(cer)
                wer_values.append(wer)
    mean_cer = sum(cer_values) / len(cer_values) if cer_values else 0.0
    mean_wer = sum(wer_values) / len(wer_values) if wer_values else 0.0
    return {"CER": mean_cer, "WER": mean_wer}

import json

def load_region_detection_ground_truth():
    # Placeholder for loading region detection ground truth data.
    # This should be replaced with actual data loading from your dataset.
    # Returns a list of dictionaries, where each dictionary represents an image and contains
    # region annotations in the format {image_path: {region_name: [x1, y1, x2, y2]}}.
    # Example: [{"image1.jpg": {"micr_line": [100, 450, 600, 500], "amount_box": [400, 100, 500, 150]}}]
    print("Loading region detection ground truth data...")
    return [
        {"image1.jpg": {"micr_line": [100, 450, 600, 500], "amount_box": [400, 100, 500, 150]}},
        {"image2.jpg": {"micr_line": [150, 400, 650, 450], "amount_box": [450, 150, 550, 200]}}
    ]

def load_text_recognition_ground_truth():
    # Placeholder for loading text recognition ground truth data.
    # This should be replaced with actual data loading from your dataset.
    # Returns a list of dictionaries, where each dictionary represents an image and contains
    # text annotations in the format {image_path: {region_name: text}}.
    # Example: [{"image1.jpg": {"micr_line": "123456789", "amount_box": "$100.00"}}]
    print("Loading text recognition ground truth data...")
    return [
        {"image1.jpg": {"micr_line": "123456789", "amount_box": "$100.00"}},
        {"image2.jpg": {"micr_line": "987654321", "amount_box": "$200.00"}}
    ]

def run_inference(image_path, region_detector_config, text_recognizer_config):
    # Placeholder for running inference with RegionDetector and TextRecognizer.
    # This should be replaced with your actual inference code.
    # Returns two lists: region_predictions and recognized_text.
    # region_predictions: A dictionary with the same format as the region detection ground truth.
    # Example: {"image1.jpg": {"micr_line": [110, 460, 590, 490], "amount_box": [410, 110, 490, 140]}}
    # recognized_text: A dictionary with the same format as the text recognition ground truth.
    # Example: {"image1.jpg": {"micr_line": "123456780", "amount_box": "$100.00"}}
    print(f"Running inference on {image_path}...")
    if image_path == "image1.jpg":
      region_predictions = {"image1.jpg": {"micr_line": [110, 460, 590, 490], "amount_box": [410, 110, 490, 140]}}
      recognized_text = {"image1.jpg": {"micr_line": "123456780", "amount_box": "$100.00"}}
    elif image_path == "image2.jpg":
      region_predictions = {"image2.jpg": {"micr_line": [140, 410, 660, 440], "amount_box": [440, 140, 560, 210]}}
      recognized_text = {"image2.jpg": {"micr_line": "987654321", "amount_box": "$200.00"}}
    else:
      region_predictions = {}
      recognized_text = {}
    return region_predictions, recognized_text



def evaluate_models(image_paths, region_detector_configs, text_recognizer_configs, output_path="evaluation_results.json"):
    print("Evaluating models...")
    region_detection_ground_truth = load_region_detection_ground_truth()
    text_recognition_ground_truth = load_text_recognition_ground_truth()
    evaluation_results = []

    for image_path in image_paths:
        image_results = []
        for region_detector_config in region_detector_configs:
            for text_recognizer_config in text_recognizer_configs:
                region_predictions, recognized_text = run_inference(image_path, region_detector_config, text_recognizer_config) 
                region_detection_metrics = calculate_region_detection_metrics([gt for gt in region_detection_ground_truth for key in gt if key in image_path], [region_predictions])
                text_recognition_metrics = calculate_text_recognition_metrics([gt for gt in text_recognition_ground_truth for key in gt if key in image_path ], [recognized_text])


                image_results.append({
                    "region_detector_config": region_detector_config,
                    "text_recognizer_config": text_recognizer_config,
                    "region_detection_metrics": region_detection_metrics,
                    "text_recognition_metrics": text_recognition_metrics
                })
        evaluation_results.append({ "image_path": image_path, "results": image_results})
    # Store results in JSON format
    with open(output_path, "w") as f:
      json.dump(evaluation_results, f, indent=4)
    print(f"Evaluation results stored in {output_path}")
