# Détection et identification d'avions sur images satellites

Projet d'Introduction à l'IA (IPSA Toulouse). Une application Python qui localise les avions
dans une image satellite et identifie leur modèle (20 classes, A1 à A20).

## Architecture

```
image satellite ──> YOLO26n (1 classe : "avion") ──> rectangles
                                                        │ découpe (crop)
                                                        ▼
                                  ResNet18 (Transfer Learning, 20 classes) ──> modèle d'avion
```

- **Détection** : YOLO26n pré-entraîné, fine-tuné sur le dataset fourni (une seule classe « avion »).
- **Classification** : ResNet18 pré-entraîné sur ImageNet, dernière couche remplacée (20 classes),
  Data Augmentation (retournements, rotation, luminosité).
- **Découpage des données** : 70 % entraînement / 15 % validation / 15 % test, fait **par image**
  (aucun décor commun entre train et test), même découpage pour le détecteur et le classifieur.
- Seul le dataset fourni est utilisé pour l'entraînement ; le Transfer Learning est le seul apport externe.

## Résultats (jeu de test)

| Brique | Résultat |
|---|---|
| Classifieur ResNet18 (vignettes découpées à la main) | 97,7 % de précision |
| Détecteur YOLO (201 images, 1165 avions) | précision 1,000 · rappel 0,998 · mAP50 0,990 · mAP50-95 0,811 |
| Bout en bout (YOLO + ResNet), nos 201 images de test | détection : précision 97,3 % / rappel 100 % · classification : 97,2 % · bout en bout : précision 94,6 % / rappel 97,2 % (détails dans `evaluation/bilan.txt`, `python etape6_evaluation.py`) |
| Corpus d'évaluation du professeur (2 511 images, 14 471 avions) | F1 global 0,816 · F1 détection 0,984 · F1 classification 0,829 (calculés par la page d'évaluation du professeur) |

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `etape2_crops.py` | découpe les avions en vignettes, séparation train / val / test |
| `etape3_entrainement.py` | entraîne le ResNet18 (Transfer Learning) |
| `etape4a_convertir_yolo.py` | convertit les annotations XML (Pascal VOC) au format YOLO |
| `etape4b_yolo.py` | entraîne le détecteur YOLO (réalisé sur Google Colab, GPU T4) |
| `etape5_pipeline.py` | pipeline complet : détection + classification + images annotées + CSV |
| `etape6_evaluation.py` | évaluation de bout en bout contre les annotations (IoU ≥ 0,5) |
| `etape7_export_json.py` | produit `predictions.json` (format de la page d'évaluation du professeur) |
| `app.py` | interface Streamlit |
| `corpus_eval_annote/` | corpus de test du professeur annoté par le modèle (images annotées + CSV) |
| `DEMO.md` | aide-mémoire pour la démonstration |
| `Dockerfile`, `requirements.txt` | déploiement |
| `sauvegardes/` | `best.pt` (YOLO) et `meilleur_modele.pth` (ResNet) |

## Utilisation

```bash
pip install -r requirements.txt

# Interface graphique
python -m streamlit run app.py

# Annoter un dossier d'images (résultats dans resultats/ + resultats.csv)
python etape5_pipeline.py "chemin/vers/images"
```

### Docker

```bash
docker build -t avions .
docker run -p 8501:8501 avions     # puis ouvrir http://localhost:8501
```

## Corpus de test annoté

Le dossier `corpus_eval_annote/` contient les images du corpus d'évaluation fourni, annotées par notre
modèle, avec le fichier CSV des détections. Le fichier `predictions.json` contient les mêmes prédictions
au format demandé par la page d'évaluation (`{"1.jpg": [{"class": "A2", "coordinates": {...}}]}`).
Ce corpus n'a jamais servi à l'entraînement ni au réglage des modèles.
