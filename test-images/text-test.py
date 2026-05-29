import pytesseract
import cv2
import pyttsx3

# Configuration Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_and_speak(image_path):
    # 1. Lecture de l'image
    img = cv2.imread(image_path)
    if img is None:
        print("Image introuvable.")
        return

    # 2. OCR en français
    text = pytesseract.image_to_string(img, lang='fra')
    print(f"Texte détecté : {text}")

    # 3. Synthèse vocale
    if text.strip():
        engine = pyttsx3.init()
        # On ajuste la vitesse
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    else:
        print("Aucun texte à lire.")

# Test
ocr_and_speak(r"../images/texte.webp")