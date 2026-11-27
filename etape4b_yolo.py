"""
Étape 4b : entraîner YOLO à repérer les avions (classe unique : "avion").

Prérequis :
  - le dossier yolo_dataset/ créé par etape4a_convertir_yolo.py,
    placé à côté de ce script ;
  - la bibliothèque ultralytics :  python -m pip install ultralytics

Fonctionne aussi bien sur ton PC que sur Google Colab : le fichier
dataset.yaml est réécrit ici avec le bon chemin (celui d'étape 4a contient
un chemin Windows qui ne marcherait pas sur Colab).

Résultats : le dossier runs/detect/train/ (courbes, exemples) et surtout
runs/detect/train/weights/best.pt = ton détecteur entraîné.
"""

import os
from pathlib import Path

from ultralytics import YOLO

# ----------------------------------------------------------------------
# 1. Paramètres
# ----------------------------------------------------------------------
DOSSIER_DATASET = Path(__file__).parent / "yolo_dataset"

# Si la décompression du zip a créé un dossier en trop (yolo_dataset/yolo_dataset),
# on descend automatiquement d'un niveau.
if (not (DOSSIER_DATASET / "images" / "train").exists()
        and (DOSSIER_DATASET / "yolo_dataset" / "images" / "train").exists()):
    DOSSIER_DATASET = DOSSIER_DATASET / "yolo_dataset"

MODELE_DEPART = "yolo26n.pt"   # petit YOLO pré-entraîné (Transfer Learning)
                               # si le téléchargement échoue : "yolo11n.pt"
NB_EPOCHS = 50                 # passages complets sur les données d'entraînement
TAILLE_IMAGE = 640             # taille de travail ; passer à 800 ou 960 si
                               # les petits avions sont ratés (plus lent)
TAILLE_LOT = 16                # images traitées en même temps
WORKERS = 0 if os.name == "nt" else 4   # 0 sous Windows (évite le plantage vu à l'étape 3)


def main():
    # ------------------------------------------------------------------
    # 2. Écrire dataset.yaml avec le chemin correct pour CET ordinateur
    # ------------------------------------------------------------------
    if not (DOSSIER_DATASET / "images" / "train").exists():
        raise SystemExit(f"Dossier introuvable : {DOSSIER_DATASET / 'images' / 'train'}\n"
                         "Place yolo_dataset/ à côté de ce script.")

    fichier_yaml = DOSSIER_DATASET / "dataset.yaml"
    fichier_yaml.write_text(
        f"path: {DOSSIER_DATASET.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: avion\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 3. Entraînement (Transfer Learning : on part d'un YOLO déjà entraîné)
    # ------------------------------------------------------------------
    modele = YOLO(MODELE_DEPART)
    modele.train(
        data=str(fichier_yaml),
        epochs=NB_EPOCHS,
        imgsz=TAILLE_IMAGE,
        batch=TAILLE_LOT,
        workers=WORKERS,
        flipud=0.5,       # retournement haut/bas : utile en vue satellite
        degrees=0,        # pas de rotation libre (les boîtes deviendraient imprécises)
    )

    # ------------------------------------------------------------------
    # 4. Examen final sur les images de TEST (jamais vues pendant l'entraînement)
    # ------------------------------------------------------------------
    metriques = modele.val(data=str(fichier_yaml), split="test")
    print("\n===== Résultats sur le jeu de test =====")
    print(f"Précision (precision) : {metriques.box.mp:.3f}")
    print(f"Rappel (recall)       : {metriques.box.mr:.3f}")
    print(f"mAP50                 : {metriques.box.map50:.3f}")
    print(f"mAP50-95              : {metriques.box.map:.3f}")


# Obligatoire sous Windows
if __name__ == "__main__":
    main()