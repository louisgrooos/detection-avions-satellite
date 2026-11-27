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
| Bout en bout (YOLO + ResNet) | voir `evaluation/bilan.txt` (`python etape6_evaluation.py`) |

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `etape2_crops.py` | découpe les avions en vignettes, séparation train / val / test |
| `etape3_entrainement.py` | entraîne le ResNet18 (Transfer Learning) |
| `etape4a_convertir_yolo.py` | convertit les annotations XML (Pascal VOC) au format YOLO |
| `etape4b_yolo.py` | entraîne le détecteur YOLO (réalisé sur Google Colab, GPU T4) |
| `etape5_pipeline.py` | pipeline complet : détection + classification + images annotées + CSV |
| `etape6_evaluation.py` | évaluation de bout en bout contre les annotations (IoU ≥ 0,5) |
| `app.py` | interface Streamlit |
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
