import os
import json
from typing import Dict
import cv2
from typing import Dict
import numpy as np
import torch
from sklearn.cluster import KMeans
import torchvision

class RegionDetector:
    def __init__(self, config_path=None, detection_params_path=None, check_type=None, model_type="dynamic"):
        self.config_path = config_path
        self.detection_params_path = detection_params_path
        self.check_type = check_type or "default"# Determine model type from config or use the provided model_type as default
        self.detection_params = self._load_detection_params()
        self.model_type = self.detection_params.get(self.check_type, {}).get("model", model_type)
        self.regions_config = self._load_regions_config()

        self.model = None
        self.yolo_model = None  # For YOLO models
        self.faster_rcnn_model = None  # For Faster R-CNN model

        # Load the model if specified
        if self.model_type == "yolov8":
            self.yolo_model = self._load_yolov8_model()
        elif self.model_type == "faster_rcnn" :
            self.faster_rcnn_model = self._load_faster_rcnn_model()
        elif self.model_type == "hybrid":
            self.yolo_model = self._load_yolo_model()
            self.faster_rcnn_model = self._load_faster_rcnn_model()
        elif self.model_type == "efficientdet":
            self.model = self._load_efficientdet_model()


        



    def _load_regions_config(self):
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {
            'micr_line': {'x1': 0.05, 'x2': 0.95, 'y1': 0.85, 'y2': 0.95},
            'amount_box': {'x1': 0.75, 'x2': 0.95, 'y1': 0.15, 'y2': 0.25},
            'payee_line': {'x1': 0.05, 'x2': 0.75, 'y1': 0.15, 'y2': 0.25},
            'date_line': {'x1': 0.75, 'x2': 0.95, 'y1': 0.05, 'y2': 0.15},
            'written_amount': {'x1': 0.05, 'x2': 0.75, 'y1': 0.35, 'y2': 0.45}
        }

    def _load_yolo_model(self):
        if self.model_type == "yolov8":
            return self._load_yolov8_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_yolov5_model(self, path):
        print(f"Loading YOLOv5 model from {path}")
        try:
            return torch.hub.load("ultralytics/yolov5", "custom", path=path, force_reload=True)
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")
            return None
        else:
            return None

    def _load_faster_rcnn_model(self):
        print(f"Loading Faster R-CNN model ")
        try:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
                # If you have a CUDA-enabled GPU, move the model to the GPU
            if torch.cuda.is_available():
               model.cuda()
            model.eval()  # Set the model to evaluation mode
            return model 
        except Exception as e:
            print(f"Error loading Faster R-CNN model: {e}")
            return None

    def _run_efficientdet_model(self, model, image):
         return self. _run_efficientdet_model(model,image)

    def _run_faster_rcnn_model(self, model, image):
        if model is None:
            return {}
        try:
            print("Running Faster R-CNN model")
            device = next(model.parameters()).device
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device) / 255.0
            
            # Faster R-CNN expects a list of tensors
            predictions = model([image_tensor])  
            
            # Extract bounding boxes, labels, and scores from the predictions
            boxes = predictions[0]['boxes'].cpu().detach().numpy()
            labels = predictions[0]['labels'].cpu().detach().numpy()
            scores = predictions[0]['scores'].cpu().detach().numpy()
            
            regions = []
            for i in range(len(boxes)):
                 box = boxes[i]
                 score = scores[i]
                 regions.append({'xmin': box[0],'ymin': box[1],'xmax': box[2],'ymax': box[3],'confidence': score})
            return regions
        except Exception as e:
            print(f"Error during Faster R-CNN inference: {e}")
            return None

    def _run_yolov5_model(self, model, image):
        if model is None:
            return {}
        try:
            print("Running YOLOv5 model")
            results = model(image)
            # Process YOLOv5 results to extract region data
            regions = {}
            for *xyxy, conf, cls in results.pred[0]:  
                x1, y1, x2, y2 = map(int, xyxy) 
                class_name = model.names[int(cls)]
                regions.append({'xmin': x1,'ymin': y1,'xmax': x2,'ymax': y2,'confidence': conf, 'class': class_name})
        

            return regions
        except Exception as e:
            print(f"Error during YOLOv5 inference: {e}")
            return {}

    def _load_yolov8_model(self):
        print(f"Loading YOLOv8 model")
        try: 
            return torch.hub.load("ultralytics/yolov8", "yolov8s",force_reload=True)
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            return None

    def _run_yolov8_model(self, model, image):
        if model is None:
            return {}
        try:
            print("Running YOLOv8 model")
            results = model(image) 
            regions = []
            for *xyxy, conf, cls in results.pred[0]: 
                x1, y1, x2, y2 = map(int, xyxy) 
                
                class_name = model.names[int(cls)]
                region_name = class_name if class_name in self.regions_config else class_name
                regions[region_name] = (x1, y1, x2, y2)
            return regions
        except Exception as e:
            print(f"Error during YOLOv8 inference: {e}")
            return {}
    
   
    def _load_efficientdet_model(self):
        print("Loading EfficientDet model")
        try:
            from effdet import get_efficientdet_b0, create_model_from_config, det_features, det_head
            from effdet.config.model_config import efficientdet_b0_config
            config = efficientdet_b0_config.copy()
            config["image_size"] = (512, 512)
            model = create_model_from_config(config, pretrained=True)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading EfficientDet model: {e}")
            return None
    
    def _run_efficientdet_model(self, model, image):
        if model is None:
            return {}
        try:
            print("Running EfficientDet model")
            from effdet.infer import get_predictions
            from effdet.utils import preprocess, postprocess
            image, ratio = preprocess(image, image_size=(512, 512))
            with torch.no_grad():
                features = det_features(model, image.unsqueeze(0))
                class_out, box_out = det_head(
                    model, features)
                detections = get_predictions(class_out, box_out, anchor_ratios=config.anchor_ratios, anchor_scales=config.anchor_scales) # Assuming config is available
                results = postprocess(detections, ratio)
            regions = {}
            if results[0] is not None:
                for x1, y1, x2, y2, score, cls in results[0]:
                    if score > 0.5:
                        regions[f"class_{int(cls)}"] = (int(x1), int(y1), int(x2), int(y2))
            return regions
        except Exception as e:
            print(f"Error during EfficientDet inference: {e}")
            return None

    def _load_detection_params(self):
        if self.detection_params_path and os.path.exists(self.detection_params_path):
            with open(self.detection_params_path, 'r') as f:
                 params = json.load(f)
            return params
        return {}


    def _compare_and_combine(self, yolo_results, faster_rcnn_results):
        print("Comparing and combining YOLOv8 and Faster R-CNN results...")

        combined_results = []

        # 1. Group detections by type (MICR, amount, etc.) - Assuming a 'class' or similar field
        yolo_by_type = {}
        faster_rcnn_by_type = {}

        for det in yolo_results:
            det_type = det.get('class')  # Adjust if your class label field is different
            if det_type:
                yolo_by_type.setdefault(det_type, []).append(det)

        for det in faster_rcnn_results:
            det_type = det.get('class')
            if det_type:
                faster_rcnn_by_type.setdefault(det_type, []).append(det)

        # 2. Iterate through expected detection types and compare
        expected_types = set(yolo_by_type.keys()).union(faster_rcnn_by_type.keys())

        for det_type in expected_types:
            yolo_dets = yolo_by_type.get(det_type, [])
            faster_rcnn_dets = faster_rcnn_by_type.get(det_type, [])

            if not yolo_dets:
                combined_results.extend(faster_rcnn_dets)
            elif not faster_rcnn_dets:
                combined_results.extend(yolo_dets)
            else:
                # Prioritize based on confidence and agreement (example)
                best_yolo = max(yolo_dets, key=lambda x: x.get('confidence', 0))
                best_faster_rcnn = max(faster_rcnn_dets, key=lambda x: x.get('confidence', 0))

                if best_yolo['confidence'] > best_faster_rcnn['confidence'] * 1.1:  # 10% confidence boost
                    combined_results.append(best_yolo)
                elif best_faster_rcnn['confidence'] > best_yolo['confidence'] * 1.1:
                    combined_results.append(best_faster_rcnn)
                else:
                    # If confidences are similar, favor Faster R-CNN (adjust as needed)
                    combined_results.append(best_faster_rcnn)

        return combined_results
    
    def detect(self, image):
        if self.model_type == "hybrid":
            return self._compare_and_combine(self._run_yolo_model(self.yolo_model, image), self._run_faster_rcnn_model(self.faster_rcnn_model, image))
        elif self.model_type == "faster_rcnn":
    def _detect_contours(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _classify_regions(self, contours, image_shape):
        h, w = image_shape
        features = []
        for contour in contours:
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            features.append([x/w, y/h, w_contour/w, h_contour/h, area/(w*h)])

        if not features:
            return {}

        kmeans = KMeans(n_clusters=min(5, len(features)), random_state=42)
        labels = kmeans.fit_predict(features)

        regions = {}
        for i, (contour, label) in enumerate(zip(contours, labels)):
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            if label == 0:  # MICR line (bottom
                regions['micr_line'] = (x, y, x + w_contour, y + h_contour)
            elif label == 1:  # Amount box (top-right)
                regions['amount_box'] = (x, y, x + w_contour, y + h_contour)
            elif label == 2:  # Payee line (left-middle)
                regions['payee_line'] = (x, y, x + w_contour, y + h_contour)
            elif label == 3:  # Date line (top-right)
                regions['date_line'] = (x, y, x + w_contour, y + h_contour)
            elif label == 4:  # Written amount (middle)
                regions['written_amount'] = (x, y, x + w_contour, y + h_contour)
        return regions

    def extract_regions(self, image, method=None) -> Dict[str, np.ndarray]:
        h, w = image.shape[:2]
        regions = {}
        method = method or self.model_type      
        
        if method == "fixed":
            for name, coords in self.regions_config.items():
                x1, x2 = int(w * coords["x1"]), int(w * coords["x2"])
                y1, y2 = int(h * coords["y1"]), int(h * coords["y2"])
                regions[name] = image[y1:y2, x1:x2]

        elif method == "yolov5":
              if self.model is None:
                  raise ValueError(f"Model {method} not loaded.")
              detected_regions = self._run_yolov5_model(self.yolo_model, image)
              for region in detected_regions:
                  name = region['class']
                  x1, y1, x2, y2 = region['xmin'], region['ymin'], region['xmax'], region['ymax']
                  
                x1, y1, x2, y2 = coords
                regions[name] = image[y1:y2, x1:x2]
        elif method == "yolov8":
            if self.model is None:
                raise ValueError(f"Model {method} not loaded.")
            detected_regions = self._run_yolov8_model(self.model, image)
            for name, coords in detected_regions.items():
                x1, y1, x2, y2 = coords
                regions[name] = image[y1:y2, x1:x2]

        elif method == "efficientdet":
             if self.model is None:
                raise ValueError(f"Model {method} not loaded.")
             detected_regions = self._run_efficientdet_model(self.yolo_model, image)
             for name, coords in detected_regions.items():
                x1, y1, x2, y2 = coords
                regions[name] = image[y1:y2, x1:x2]
        elif method == "faster_rcnn" :
            detected_regions = self._run_faster_rcnn_model(self.faster_rcnn_model, image)
            for region in detected_regions:
                 x1, y1, x2, y2 = region['xmin'], region['ymin'], region['xmax'], region['ymax']

                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                 name = "faster_rcnn_detected"
                
                 
                 regions[name] = image[y1:y2, x1:y2]
        
        elif method == "efficientdet":
            detected_regions = self._run_efficientdet_model(self.model, image)
            pass
        elif method != "dynamic":  # for later other models
            if self.model is None:
                raise ValueError(f"Model {self.model_type} not loaded.")
            detected_regions = self._run_yolov5_model(self.model, image)
            for name, coords in detected_regions.items():
                x1, y1, x2, y2 = coords
                regions[name] = image[y1:y2, x1:x2]
        else:  # dynamic method == 'dynamic'
            contours = self._detect_contours(image)
            detected_regions = self._classify_regions(contours, (h, w))
            for name in self.regions_config:
                if name in detected_regions:
                    x1, y1, x2, y2 = detected_regions[name]
                    regions[name] = image[y1:y2, x1:x2]
                else:
                    x1, x2 = int(w * self.regions_config[name]["x1"]), int(w * self.regions_config[name]["x2"])
                    y1, y2 = int(h * self.regions_config[name]["y1"]), int(h * self.regions_config[name]["y2"])
                    regions[name] = image[y1:y2, x1:x2]

        return regions
        else:
            return self._run_faster_rcnn_model(self.faster_rcnn_model, image)
