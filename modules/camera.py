import cv2
from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT

class CameraManager:
    def __init__(self):
        """
        Gère la capture vidéo
        """
        self.cap = cv2.VideoCapture(CAMERA_ID)
        
        # Configurer la résolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            raise Exception("Impossible d'ouvrir la caméra")
    
    def get_frame(self):
        """
        Capture une image
        """
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def release(self):
        """
        Libère la caméra
        """
        self.cap.release()