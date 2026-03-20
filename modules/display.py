import cv2

class DisplayManager:
    @staticmethod
    def show(frame, objects, alerts):
        """
        Affiche l'image avec les objets détectés
        """
        # Copie pour l'affichage
        display = frame.copy()
        
        # Dessiner chaque objet détecté
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            class_name = obj['class']
            conf = obj['confidence']
            distance = obj.get('distance', '?')
            
            # Choisir la couleur selon le danger
            color = (0, 255, 0)  # Vert par défaut
            for alert in alerts:
                if alert['type'] == class_name:
                    color = (0, 0, 255)  # Rouge si danger
            
            # Dessiner le rectangle
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Texte
            label = f"{class_name} {conf:.2f} {distance}m"
            cv2.putText(display, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Afficher le nombre d'objets
        cv2.putText(display, f"Objets: {len(objects)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Afficher l'alerte en cours
        if alerts:
            alert_text = alerts[0]['message']
            cv2.putText(display, f"⚠️ {alert_text}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Montrer l'image
        cv2.imshow('ClearVision - Prototype', display)