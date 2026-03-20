import time
from config import DANGER_ZONES, ALERT_COOLDOWN

class DecisionEngine:
    def __init__(self):
        """
        Décide s'il faut alerter l'utilisateur
        """
        self.last_alert_time = 0
        self.last_alert_type = None
        
    def analyze(self, detected_objects):
        """
        Analyse les objets détectés et retourne les alertes nécessaires
        """
        alerts = []
        current_time = time.time()
        
        # Vérifier chaque objet
        for obj in detected_objects:
            class_name = obj['class']
            distance = obj.get('distance', 999)
            
            # Seuil de danger pour cette classe
            danger_threshold = DANGER_ZONES.get(class_name, 2.0)  # 2m par défaut
            
            if distance < danger_threshold:
                # Calculer la direction
                center_x = obj['center'][0]
                frame_center = 320  # À adapter selon config
                
                if center_x < frame_center - 100:
                    direction = "gauche"
                elif center_x > frame_center + 100:
                    direction = "droite"
                else:
                    direction = "devant"
                
                # Message d'alerte
                alert = {
                    'type': class_name,
                    'distance': distance,
                    'direction': direction,
                    'severity': 'high' if distance < danger_threshold/2 else 'medium',
                    'message': f"{class_name} à {distance}m sur la {direction}"
                }
                
                alerts.append(alert)
        
        # Filtrer les alertes trop fréquentes
        if current_time - self.last_alert_time < ALERT_COOLDOWN:
            return []  # Trop tôt pour une nouvelle alerte
        
        # Priorité: l'objet le plus proche
        if alerts:
            # Trier par distance
            alerts.sort(key=lambda x: x['distance'])
            self.last_alert_time = current_time
            return [alerts[0]]  # Retourne seulement la plus urgente
        
        return []