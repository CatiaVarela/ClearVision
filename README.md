# ClearVision

Détection d'objets et d'obstacles sur **photo**, **vidéo** ou **webcam**, en local (modèles YOLO). Les éléments reconnus sont entourés d'un rectangle bleu avec une étiquette en français.

**Prérequis :** Python 3.10+, connexion internet au premier lancement (téléchargement des modèles).

---

## Installation

```powershell
pip install -r requirements.txt
```

Placez vos fichiers de test dans `images/` (photos) ou `videos/` (vidéos).

---

## Commandes du projet

Toutes les commandes ci-dessous se lancent depuis la **racine du projet**, sauf indication contraire.

### Installation et tests

| Commande | Description |
|----------|-------------|
| `pip install -r requirements.txt` | Installe les dépendances Python (PyTorch, OpenCV, Ultralytics, etc.). |
| `python -m pytest tests/ -v` | Lance les tests automatisés du projet. |

---

### Analyser une photo (commande principale)

| Commande | Description |
|----------|-------------|
| `python detect_image.py images/ma-photo.jpg` | Analyse une image, affiche le résultat et sauvegarde `ma-photo_detected.jpg`. |
| `python detect_image.py images/ma-photo.jpg --fast` | Mode rapide : COCO uniquement, sans YOLO-World (animaux/plantes rares en moins). |
| `python detect_image.py images/ma-photo.jpg --no-gui` | Même analyse sans fenêtre OpenCV (utile en SSH ou si l'affichage plante). |
| `python detect_image.py images/ma-photo.jpg --output sortie.jpg` | Enregistre le résultat sous un nom de fichier personnalisé. |
| `python detect_image.py images/ma-photo.jpg --json resultats.json` | Exporte les détections (nom, position, score) dans un fichier JSON. |

---

### Interface web et API

| Commande | Description |
|----------|-------------|
| `uvicorn api.main:app --reload` | Démarre le serveur web local → ouvrir [http://localhost:8000](http://localhost:8000) pour envoyer une image par glisser-déposer. |
| `curl -X POST -F "file=@images/ma-photo.jpg" http://localhost:8000/detect` | Envoie une image à l'API et récupère les détections en JSON (serveur déjà lancé). |

---

### Webcam

| Commande | Description |
|----------|-------------|
| `cd test-cam` puis `python cam-voice.py` | Webcam en direct : obstacles proches, distance estimée, annonce vocale. Touche **q** pour quitter. |

---

### Vidéo

| Commande | Description |
|----------|-------------|
| `cd test-videos` puis `python video-test.py --video ../videos/ma-video.mp4` | Analyse une vidéo et exporte `ma-video_yoloworld.mp4` avec rectangles bleus. |
| `python video-test.py --video ../videos/ma-video.mp4 --no-gui` | Traitement vidéo sans prévisualisation à l'écran. |
| `python video-test.py --video ../videos/ma-video.mp4 --fast` | Mode rapide sans YOLO-World. |
| `python video-test.py --video ../videos/ma-video.mp4 --voice` | Ajoute une piste vocale dans le MP4 (obstacles proches annoncés). |
| `python video-test.py --video ../videos/ma-video.mp4 --max-distance 15` | Affiche les obstacles jusqu'à 15 m (utile en extérieur / rue). |

---

### Apprendre un nouvel objet

| Commande | Description |
|----------|-------------|
| `python scripts/learn_object.py "banc public" --count 10 --train` | Télécharge ~10 images, les annote, entraîne le modèle sur cet objet. |
| `python scripts/learn_object.py "fire hydrant" --search-en --count 10 --train` | Recherche en anglais (souvent de meilleurs résultats sur Internet). |
| `python scripts/learn_object.py "fire hydrant" --label-fr "borne incendie" --train` | Définit le libellé français affiché à l'écran. |
| `python scripts/learn_object.py "fire hydrant" --skip-download --train` | Réutilise les images déjà dans `datasets/custom/sources/` et relance l'entraînement. |
| `python scripts/learn_object.py "banc public" --epochs 50 --train` | Entraînement plus long (50 époques au lieu de 40). |

Après entraînement, activer le modèle dans `config.py` :

```python
USE_CUSTOM_MODEL = True
```

| Commande | Description |
|----------|-------------|
| `python scripts/train_custom_yolo.py` | Entraîne le modèle custom à partir du dataset déjà préparé (sans télécharger d'images). |
| `python scripts/export_model.py` | Exporte le modèle entraîné (`best.pt`) au format ONNX. |

---

### Scripts utilitaires (développeurs)

| Commande | Description |
|----------|-------------|
| `python scripts/benchmark_models.py` | Compare la vitesse des modèles YOLO (`n`, `s`, `m`) sur les images du dossier `images/`. |
| `python scripts/evaluate_detection.py` | Génère un rapport `evaluation_report.md` à partir des images de test dans `tests/fixtures/`. |
| `python test-images/yolo-world-test.py --image images/ma-photo.jpg` | Ancien script image (alternative à `detect_image.py`). |
| `python test-images/yolo-world-test.py --image images/ma-photo.jpg --fast --no-gui` | Même script, mode rapide sans fenêtre. |

---

## Fichiers utiles

| Fichier | Rôle |
|---------|------|
| `config.py` | Seuils de détection, modèles, `USE_CUSTOM_MODEL` |
| `vocabulary.py` | Liste des animaux, plantes et objets recherchés par YOLO-World |
| `datasets/README.md` | Détails sur le dataset et l'apprentissage d'objets |
| `PLAN_RECONNAISSANCE_IMAGE.md` | Plan complet du projet |

---

*ClearVision — détection locale, rectangles bleus, étiquettes en français.*
