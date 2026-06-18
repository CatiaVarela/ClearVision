# Images de test ClearVision

Placez ici vos photos de test pour la reconnaissance d'objets.

## Scènes recommandées

| Scène | Contenu attendu | Objectif |
|-------|-----------------|----------|
| Parc | personnes, arbres, haies, bancs | végétation + objets COCO |
| Salon | meubles, plantes en pot, animaux domestiques | objets courants |
| Rue | voitures, piétons, panneaux | détection urbaine |
| Nature | animaux sauvages, fleurs, arbres | YOLO-World élargi |

## Utilisation

```powershell
python detect_image.py images/ma-photo.jpg
```

Les fichiers image ne sont pas versionnés (voir `.gitignore`).
