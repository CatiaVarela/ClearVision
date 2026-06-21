# Datasets ClearVision

## Apprendre un nouvel objet (automatique)

ClearVision peut **télécharger des images sur Internet**, les **annoter** et **entraîner** le modèle :

```powershell
pip install -r requirements.txt
python scripts/learn_object.py "banc public" --count 10 --train
```

### Ce que fait la commande

1. Cherche ~10 images sur Internet (`datasets/custom/sources/banc_public/`)
2. Les annote automatiquement avec YOLO-World
3. Les ajoute au dataset (`images/train`, `images/val`, `labels/...`)
4. Entraîne le modèle YOLO → `datasets/custom/runs/clearvision_custom/weights/best.pt`

### Activer le modèle appris

Dans `config.py` :

```python
USE_CUSTOM_MODEL = True
```

Le modèle custom **s'ajoute** au modèle COCO standard (il ne le remplace pas).

### Options utiles

```powershell
# Nom en anglais (meilleurs résultats parfois)
python scripts/learn_object.py "fire hydrant" --search-en --count 10 --train

# Libellé français affiché
python scripts/learn_object.py "trash can" --label-fr "poubelle" --train

# Ré-entraîner sans re-télécharger
python scripts/learn_object.py "banc public" --skip-download --train
```

---

## Structure attendue (format YOLO)

```
datasets/custom/
├── dataset.yaml
├── learned_classes.yaml
├── sources/              ← images brutes téléchargées
│   └── banc_public/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Annotation manuelle (optionnel)

Si l'annotation auto échoue, utilisez :

- [Roboflow](https://roboflow.com) — export format YOLOv8
- [CVAT](https://cvat.ai)
- [Label Studio](https://labelstud.io)

Format YOLO : `class_id x_center y_center width height` (coordonnées normalisées 0–1)

## Licence

Les images web sont pour usage éducatif / projet scolaire. Vérifiez les droits avant un déploiement public.

## Entraînement seul

```powershell
python scripts/train_custom_yolo.py
```
