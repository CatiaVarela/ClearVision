# ClearVision
Détection d'obstacles par IA (via webcam ou vidéo) avec estimation des distances et retour vocal pour les personnes malvoyantes.

## Installation

```powershell
pip install -r requirements.txt
```

La voix dans le MP4 utilise `imageio-ffmpeg` (ffmpeg inclus, installé via `pip install -r requirements.txt`).

## Webcam + voix (≤ 4 m)

```powershell
cd test-cam
python cam-voice.py
```

## Vidéo + voix enregistrée dans le MP4

```powershell
cd test-videos
python video-test.py --video ../videos/38805-418875307.mp4 --voice --no-gui
```

La sortie `*_yoloworld.mp4` contient l'image annotée **et** la piste audio des annonces.

## Réglages

Fichier `clearvision_voice.py` : `MAX_DISTANCE_M`, `FOCAL_LENGTH_FACTOR`, `VOICE_LABELS`.
