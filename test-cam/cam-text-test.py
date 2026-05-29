import cv2
import pytesseract
import pyttsx3

# Configuration Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialisation de la voix
engine = pyttsx3.init()
engine.setProperty('rate', 160)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success: break

    #frame = cv2.flip(frame, 1)

    # --- ÉTAPE CRUCIALE ---
    # On crée une copie "propre" sans les dessins/textes de l'interface
    frame_clean = frame.copy()

    # On dessine l'instruction UNIQUEMENT sur l'image qui sera affichée (frame)
    cv2.putText(frame, "Appuyez sur ESPACE pour lire", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Lecteur Intelligent", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        print("Analyse du texte en cours...")

        # On fait l'OCR sur frame_clean (celle qui n'a pas le texte vert)
        gray = cv2.cvtColor(frame_clean, cv2.COLOR_BGR2GRAY)

        # Petit prétraitement pour ignorer les bruits
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        text = pytesseract.image_to_string(gray, lang='fra')

        if text.strip():
            print(f"Texte détecté : {text}")
            engine.say(text)
            engine.runAndWait()
        else:
            print("Désolé, je n'ai rien pu lire.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()