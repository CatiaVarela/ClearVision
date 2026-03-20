import numpy as np

class DistanceEstimator:
    def __init__(self, frame_width=640, frame_height=480):
        """
        Estime la distance des objets détectés
        Méthode simplifiée: utilise la hauteur de l'objet dans l'image
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Taille moyenne connue des objets (en mètres)
        self.known_sizes = {
            'person': 1.7,      # Taille moyenne d'une personne
            'car': 1.5,          # Hauteur d'une voiture
            'truck': 3.0,        # Hauteur camion
            'bus': 3.0,
            'bicycle': 1.0,       # Hauteur vélo
            'motorcycle': 1.2,
            'dog': 0.5,
            'cat': 0.3,
            'chair': 0.8,
            'table': 0.7,
            'tree': 2.0,          # Branche en hauteur
            'pole': 2.0,          # Poteau
            'wall': 2.0,          # Mur (hauteur standard)
            'door': 2.0
        }
        
        # Focale approximative (à calibrer selon ta webcam)
        # Formule: distance = (taille_réelle * focale) / taille_dans_image
        self.focal_length = 500  # À ajuster expérimentalement
    
    def estimate(self, detected_objects):
        """
        Ajoute une estimation de distance à chaque objet
        """
        for obj in detected_objects:
            class_name = obj['class']
            
            # Hauteur de l'objet dans l'image (pixels)
            x1, y1, x2, y2 = obj['bbox']
            height_pixels = y2 - y1
            
            if height_pixels > 0:
                # Taille réelle estimée
                real_size = self.known_sizes.get(class_name, 0.5)  # 0.5m par défaut
                
                # Distance = (taille_réelle * focale) / taille_pixels
                distance = (real_size * self.focal_length) / height_pixels
                obj['distance'] = round(distance, 2)
            else:
                obj['distance'] = 999  # Infini
            
        return detected_objects