"""
Étape 5 : le pipeline complet.

Pour chaque image satellite :
  1. YOLO (best.pt) trouve les avions                 -> "OÙ sont-ils ?"
  2. on découpe chaque rectangle trouvé (vignette)
  3. le ResNet (meilleur_modele.pth) classe la vignette -> "QUEL avion est-ce ?"
  4. on dessine le résultat sur l'image et on l'enregistre

Sorties (dans le dossier "resultats") :
  - les images annotées ;
  - resultats.csv : une ligne par avion détecté (position, classe, confiances).

Prérequis (dans le .venv) :
    python -m pip install ultralytics opencv-python pillow

Fichiers attendus, par défaut :
    sauvegardes/best.pt                  (détecteur YOLO, téléchargé depuis Colab)
    sauvegardes/meilleur_modele.pth      (classifieur ResNet de l'étape 3)
"""

from collections import Counter
from pathlib import Path
import csv
import sys
import time

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# ----------------------------------------------------------------------
# 1. Paramètres : à adapter si besoin
# ----------------------------------------------------------------------
DOSSIER_SCRIPT = Path(__file__).parent

CHEMIN_YOLO = DOSSIER_SCRIPT / "sauvegardes" / "best.pt"
CHEMIN_RESNET = DOSSIER_SCRIPT / "sauvegardes" / "meilleur_modele.pth"

# Images à analyser. Par défaut : les images de TEST de ton dataset.
# (Pour un autre dossier : DOSSIER_IMAGES = Path(r"C:\chemin\vers\les\images"))
DOSSIER_IMAGES = DOSSIER_SCRIPT / "yolo_dataset" / "images" / "test"
DOSSIER_SORTIE = DOSSIER_SCRIPT / "resultats"

NB_IMAGES_MAX = 10          # pour un premier essai rapide ; mets None pour tout traiter
SEUIL_DETECTION = 0.25      # confiance minimale de YOLO pour garder un rectangle
TAILLE_IMAGE_YOLO = 640     # même taille que pendant l'entraînement de YOLO

# Usage en ligne de commande (traite alors TOUTES les images du dossier) :
#   python etape5_pipeline.py "C:\\chemin\\images" ["C:\\chemin\\sortie"]
if __name__ == "__main__" and len(sys.argv) >= 2:
    DOSSIER_IMAGES = Path(sys.argv[1])
    NB_IMAGES_MAX = None
    if len(sys.argv) >= 3:
        DOSSIER_SORTIE = Path(sys.argv[2])

device = "cuda" if torch.cuda.is_available() else "cpu"

# Préparation des vignettes : EXACTEMENT la même qu'à l'étape 3 (val/test)
transfo_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ----------------------------------------------------------------------
# 2. Charger les deux modèles
# ----------------------------------------------------------------------
def charger_resnet(chemin):
    """Recrée le ResNet18 de l'étape 3 et y charge les poids sauvegardés."""
    sauvegarde = torch.load(chemin, map_location=device)
    classes = sauvegarde["classes"]                       # ['A1', 'A10', ...]
    modele = models.resnet18(weights=None)                # pas besoin de re-télécharger ImageNet
    modele.fc = nn.Linear(modele.fc.in_features, len(classes))
    modele.load_state_dict(sauvegarde["poids"])
    return modele.to(device).eval(), classes


# ----------------------------------------------------------------------
# 3. Fonctions du pipeline
# ----------------------------------------------------------------------
def recadrer(boite, largeur, hauteur):
    """Coordonnées entières, gardées DANS l'image. Renvoie None si la boîte est vide."""
    x1, y1, x2, y2 = (int(round(v)) for v in boite)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(largeur, x2), min(hauteur, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def analyser_image(chemin_image, yolo, resnet, classes):
    """Renvoie la liste des avions trouvés dans une image (un dictionnaire par avion)."""
    image = Image.open(chemin_image).convert("RGB")
    largeur, hauteur = image.size

    # --- Étape A : YOLO trouve les rectangles ---
    resultat = yolo.predict(
        str(chemin_image), conf=SEUIL_DETECTION, imgsz=TAILLE_IMAGE_YOLO, verbose=False
    )[0]
    boites = resultat.boxes.xyxy.cpu().tolist()       # [x1, y1, x2, y2] en pixels
    confiances = resultat.boxes.conf.cpu().tolist()   # confiance de la détection

    # --- Étape B : on découpe chaque rectangle ---
    vignettes, positions, confs_detection = [], [], []
    for boite, conf in zip(boites, confiances):
        coords = recadrer(boite, largeur, hauteur)
        if coords is None:
            continue
        vignettes.append(transfo_resnet(image.crop(coords)))
        positions.append(coords)
        confs_detection.append(conf)

    if not vignettes:
        return []

    # --- Étape C : le ResNet classe toutes les vignettes d'un coup ---
    lot = torch.stack(vignettes).to(device)
    with torch.no_grad():
        probabilites = torch.softmax(resnet(lot), dim=1)    # probabilité de chaque classe
        confs_classif, indices = probabilites.max(dim=1)    # la meilleure classe

    detections = []
    for coords, conf_det, conf_cls, indice in zip(
        positions, confs_detection, confs_classif.tolist(), indices.tolist()
    ):
        detections.append({
            "x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3],
            "confiance_detection": conf_det,
            "classe": classes[indice],
            "confiance_classification": conf_cls,
        })
    return detections


def dessiner(chemin_image, detections, chemin_sortie):
    """Dessine les rectangles et les étiquettes (classe + confiance) sur l'image."""
    image = cv2.imread(str(chemin_image))
    for d in detections:
        cv2.rectangle(image, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 0, 255), 2)
        texte = f'{d["classe"]} {d["confiance_classification"]:.0%}'
        y_texte = d["y1"] - 6 if d["y1"] > 20 else d["y2"] + 16
        cv2.putText(image, texte, (d["x1"], y_texte),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite(str(chemin_sortie), image)


def ecrire_csv(chemin_csv, lignes):
    """Une ligne par avion. Séparateur ';' pour s'ouvrir directement dans Excel (FR)."""
    colonnes = ["image", "x1", "y1", "x2", "y2",
                "confiance_detection", "classe", "confiance_classification"]
    with open(chemin_csv, "w", newline="", encoding="utf-8-sig") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=colonnes, delimiter=";")
        ecrivain.writeheader()
        for ligne in lignes:
            ecrivain.writerow({
                **ligne,
                "confiance_detection": f'{ligne["confiance_detection"]:.3f}',
                "confiance_classification": f'{ligne["confiance_classification"]:.3f}',
            })


# ----------------------------------------------------------------------
# 4. Programme principal
# ----------------------------------------------------------------------
def main():
    for chemin in (CHEMIN_YOLO, CHEMIN_RESNET):
        if not chemin.exists():
            raise SystemExit(f"Fichier introuvable : {chemin}\n"
                             "Vérifie les chemins CHEMIN_YOLO / CHEMIN_RESNET en haut du script.")
    if not DOSSIER_IMAGES.exists():
        raise SystemExit(f"Dossier d'images introuvable : {DOSSIER_IMAGES}")

    print(f"Calcul sur : {device}")
    yolo = YOLO(str(CHEMIN_YOLO))
    resnet, classes = charger_resnet(CHEMIN_RESNET)
    print(f"Modèles chargés ({len(classes)} classes d'avions)")

    images = sorted(p for p in DOSSIER_IMAGES.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if NB_IMAGES_MAX is not None:
        images = images[:NB_IMAGES_MAX]
    print(f"{len(images)} images à analyser dans {DOSSIER_IMAGES}\n")

    DOSSIER_SORTIE.mkdir(exist_ok=True)
    lignes_csv = []
    compteur_classes = Counter()
    debut = time.time()

    for numero, chemin_image in enumerate(images, start=1):
        detections = analyser_image(chemin_image, yolo, resnet, classes)
        dessiner(chemin_image, detections, DOSSIER_SORTIE / chemin_image.name)

        for d in detections:
            lignes_csv.append({"image": chemin_image.name, **d})
            compteur_classes[d["classe"]] += 1
        print(f"[{numero}/{len(images)}] {chemin_image.name} : {len(detections)} avion(s)")

    ecrire_csv(DOSSIER_SORTIE / "resultats.csv", lignes_csv)

    duree = time.time() - debut
    print(f"\nTerminé en {duree:.1f} s ({duree / max(1, len(images)):.2f} s par image)")
    print(f"{len(lignes_csv)} avions détectés au total. Répartition :")
    for classe, nombre in sorted(compteur_classes.items(), key=lambda x: (len(x[0]), x[0])):
        print(f"  {classe:<5} {nombre}")
    print(f"\nImages annotées et resultats.csv dans : {DOSSIER_SORTIE}")


if __name__ == "__main__":
    main()
