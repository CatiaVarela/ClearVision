import cv2
from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="models/yolov8n.pt", conf_threshold=0.5):
        """
        Initialise le détecteur YOLO
        """
        print("Chargement du modèle YOLO...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        print("Modèle chargé !")
        
        # Classes que YOLO peut détecter (version simplifiée)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def detect(self, frame):
        """
        Détecte les objets dans une image
        Retourne: liste d'objets avec [classe, confiance, boîte]
        """
        # YOLO attend une image RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Détection
        results = self.model(frame_rgb, verbose=False)[0]
        
        detected_objects = []
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
                
            # Coordonnées de la boîte
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Classe
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            
            detected_objects.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'center': [int((x1+x2)/2), int((y1+y2)/2)]
            })
        
        return detected_objects