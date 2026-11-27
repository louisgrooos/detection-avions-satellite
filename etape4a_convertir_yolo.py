"""
Étape 4a : convertir le dataset au format YOLO (pour la détection).

YOLO attend :
    yolo_dataset/
    ├── images/train, images/val, images/test     (les photos)
    ├── labels/train, labels/val, labels/test     (un fichier .txt par photo)
    └── dataset.yaml                              (le "mode d'emploi" du dataset)

Chaque ligne d'un fichier .txt décrit UN avion :
    0 x_centre y_centre largeur hauteur
avec des valeurs entre 0 et 1 (en proportion de la taille de l'image).

Ici, YOLO ne cherche que "avion" (classe 0). C'est ton ResNet qui donnera
ensuite le modèle précis (A1, A10...).

Le découpage train/val/test est IDENTIQUE à celui de l'étape 2
(même graine, même méthode) : une image de test pour le classifieur
reste une image de test pour le détecteur.

Aucune nouvelle installation : OpenCV suffit.
"""

from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

import cv2

# ----------------------------------------------------------------------
# 1. Paramètres : à adapter à TON ordinateur (comme à l'étape 2)
# ----------------------------------------------------------------------
DOSSIER = Path(r"C:\Users\louis\Documents\Cours\IA\military-aircraft-recognition-dataset.train\student-dataset 2")
DOSSIER_IMAGES = DOSSIER / "images"
DOSSIER_ANNOTATIONS = DOSSIER / "annotations"

SORTIE = Path(__file__).parent / "yolo_dataset"

PROPORTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
GRAINE = 42   # NE PAS CHANGER : c'est la même qu'à l'étape 2


# ----------------------------------------------------------------------
# 2. Fonctions utiles
# ----------------------------------------------------------------------
def lire_boites(chemin_xml):
    """Renvoie la liste des boîtes (xmin, ymin, xmax, ymax) d'un XML."""
    racine = ET.parse(chemin_xml).getroot()
    boites = []
    for objet in racine.findall("object"):
        b = objet.find("bndbox")
        boites.append((
            int(float(b.find("xmin").text)),
            int(float(b.find("ymin").text)),
            int(float(b.find("xmax").text)),
            int(float(b.find("ymax").text)),
        ))
    return boites


def trouver_image(chemin_xml):
    for extension in (".jpg", ".JPG", ".jpeg", ".png"):
        chemin = DOSSIER_IMAGES / (chemin_xml.stem + extension)
        if chemin.exists():
            return chemin
    return None


# ----------------------------------------------------------------------
# 3. Même découpage qu'à l'étape 2
# ----------------------------------------------------------------------
fichiers_xml = sorted(DOSSIER_ANNOTATIONS.glob("*.xml"))
print(f"{len(fichiers_xml)} fichiers d'annotations trouvés")
if not fichiers_xml:
    raise SystemExit("Aucun XML trouvé : vérifie le chemin DOSSIER.")

random.seed(GRAINE)
random.shuffle(fichiers_xml)

nb_train = int(len(fichiers_xml) * PROPORTIONS["train"])
nb_val = int(len(fichiers_xml) * PROPORTIONS["val"])

decoupage = {
    "train": fichiers_xml[:nb_train],
    "val": fichiers_xml[nb_train:nb_train + nb_val],
    "test": fichiers_xml[nb_train + nb_val:],
}


# ----------------------------------------------------------------------
# 4. Copier les images et écrire les fichiers d'étiquettes
# ----------------------------------------------------------------------
nb_avions = 0
ignores = 0

for nom_split, liste_xml in decoupage.items():
    (SORTIE / "images" / nom_split).mkdir(parents=True, exist_ok=True)
    (SORTIE / "labels" / nom_split).mkdir(parents=True, exist_ok=True)

    for chemin_xml in liste_xml:
        chemin_image = trouver_image(chemin_xml)
        if chemin_image is None:
            print(f"Image introuvable pour {chemin_xml.name}, ignorée")
            continue

        image = cv2.imread(str(chemin_image))
        if image is None:
            print(f"Image illisible : {chemin_image.name}, ignorée")
            continue
        hauteur, largeur = image.shape[:2]

        lignes = []
        for xmin, ymin, xmax, ymax in lire_boites(chemin_xml):
            # On garde la boîte DANS l'image
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(largeur, xmax), min(hauteur, ymax)
            if xmax <= xmin or ymax <= ymin:
                ignores += 1
                continue

            # Format YOLO : centre et taille, en proportion de l'image (0 à 1)
            x_centre = (xmin + xmax) / 2 / largeur
            y_centre = (ymin + ymax) / 2 / hauteur
            w = (xmax - xmin) / largeur
            h = (ymax - ymin) / hauteur
            lignes.append(f"0 {x_centre:.6f} {y_centre:.6f} {w:.6f} {h:.6f}")

        shutil.copy(chemin_image, SORTIE / "images" / nom_split / chemin_image.name)
        (SORTIE / "labels" / nom_split / f"{chemin_image.stem}.txt").write_text(
            "\n".join(lignes), encoding="utf-8"
        )
        nb_avions += len(lignes)

    print(f"  {nom_split:<5} : {len(liste_xml)} images")


# ----------------------------------------------------------------------
# 5. Le fichier dataset.yaml (le "mode d'emploi" lu par YOLO)
# ----------------------------------------------------------------------
yaml = f"""path: {SORTIE.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: avion
"""
(SORTIE / "dataset.yaml").write_text(yaml, encoding="utf-8")

print(f"\n{nb_avions} avions écrits, {ignores} boîtes invalides ignorées")
print(f"Dataset YOLO prêt dans : {SORTIE}")
