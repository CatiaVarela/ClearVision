# Datasets ClearVision

## Structure attendue (format YOLO)

```
datasets/custom/
├── dataset.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Annotation

Utilisez un outil externe :

- [Roboflow](https://roboflow.com) — export format YOLOv8
- [CVAT](https://cvat.ai)
- [Label Studio](https://labelstud.io)

## Convention de nommage

- Une image `photo001.jpg` → label `photo001.txt` (même nom, extension `.txt`)
- Format YOLO : `class_id x_center y_center width height` (coordonnées normalisées 0–1)

## Licence

Documentez la source et la licence de chaque image avant entraînement.

## Entraînement

```powershell
python scripts/train_custom_yolo.py
```

Puis activez le modèle dans `config.py` : `USE_CUSTOM_MODEL = True`.
