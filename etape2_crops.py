"""
Étape 2 : préparer les données pour la classification.

Ce script :
  1. lit toutes les annotations XML du dataset ;
  2. sépare les IMAGES en train / val / test (70 % / 15 % / 15 %) ;
  3. découpe chaque avion en vignette et la range dans un dossier par classe.

Résultat :
    crops/
    ├── train/A1/xxx_0.jpg, xxx_1.jpg, ...
    ├── train/A2/...
    ├── val/...
    └── test/...

Prérequis : python -m pip install opencv-python
"""

from pathlib import Path
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import cv2

# ----------------------------------------------------------------------
# 1. Paramètres : à adapter à TON ordinateur
# ----------------------------------------------------------------------
DOSSIER = Path(r"C:\Users\louis\Documents\Cours\IA\military-aircraft-recognition-dataset.train\student-dataset 2")
DOSSIER_IMAGES = DOSSIER / "images"
DOSSIER_ANNOTATIONS = DOSSIER / "annotations"

# Les vignettes seront créées dans un dossier "crops" à côté de ce script
SORTIE = Path(__file__).parent / "crops"

PROPORTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
GRAINE = 42   # fixe le hasard : tu obtiens toujours le même découpage


# ----------------------------------------------------------------------
# 2. Fonctions utiles
# ----------------------------------------------------------------------
def lire_annotation(chemin_xml):
    """Renvoie la liste des avions d'un XML (classe + coordonnées de la boîte)."""
    racine = ET.parse(chemin_xml).getroot()
    avions = []
    for objet in racine.findall("object"):
        boite = objet.find("bndbox")
        avions.append({
            "classe": objet.find("name").text,
            "xmin": int(float(boite.find("xmin").text)),
            "ymin": int(float(boite.find("ymin").text)),
            "xmax": int(float(boite.find("xmax").text)),
            "ymax": int(float(boite.find("ymax").text)),
        })
    return avions


def trouver_image(chemin_xml):
    """Retrouve l'image qui porte le même nom que le XML."""
    for extension in (".jpg", ".JPG", ".jpeg", ".png"):
        chemin = DOSSIER_IMAGES / (chemin_xml.stem + extension)
        if chemin.exists():
            return chemin
    return None


# ----------------------------------------------------------------------
# 3. Séparer les IMAGES (et non les avions) en train / val / test
#    -> évite qu'un même décor se retrouve dans train ET dans test
# ----------------------------------------------------------------------
fichiers_xml = sorted(DOSSIER_ANNOTATIONS.glob("*.xml"))
print(f"{len(fichiers_xml)} fichiers d'annotations trouvés")
if not fichiers_xml:
    raise SystemExit("Aucun XML trouvé : vérifie le chemin DOSSIER.")

random.seed(GRAINE)
random.shuffle(fichiers_xml)               # on mélange les images

nb_train = int(len(fichiers_xml) * PROPORTIONS["train"])
nb_val = int(len(fichiers_xml) * PROPORTIONS["val"])

decoupage = {
    "train": fichiers_xml[:nb_train],
    "val": fichiers_xml[nb_train:nb_train + nb_val],
    "test": fichiers_xml[nb_train + nb_val:],
}
for nom, liste in decoupage.items():
    print(f"  {nom:<5} : {len(liste)} images")


# ----------------------------------------------------------------------
# 4. Découper et sauvegarder les vignettes
# ----------------------------------------------------------------------
compteurs = defaultdict(Counter)   # compteurs["train"]["A10"] = nombre de vignettes
ignorees = 0

for nom_split, liste_xml in decoupage.items():
    for i, chemin_xml in enumerate(liste_xml, start=1):

        chemin_image = trouver_image(chemin_xml)
        if chemin_image is None:
            print(f"Image introuvable pour {chemin_xml.name}, ignorée")
            continue

        image = cv2.imread(str(chemin_image))      # on garde le BGR : cv2.imwrite l'attend
        if image is None:
            print(f"Image illisible : {chemin_image.name}, ignorée")
            continue
        hauteur, largeur = image.shape[:2]

        for numero, avion in enumerate(lire_annotation(chemin_xml)):
            # On s'assure que la boîte reste DANS l'image
            xmin = max(0, avion["xmin"])
            ymin = max(0, avion["ymin"])
            xmax = min(largeur, avion["xmax"])
            ymax = min(hauteur, avion["ymax"])

            if xmax <= xmin or ymax <= ymin:       # boîte vide ou incohérente
                ignorees += 1
                continue

            vignette = image[ymin:ymax, xmin:xmax]

            dossier_classe = SORTIE / nom_split / avion["classe"]
            dossier_classe.mkdir(parents=True, exist_ok=True)
            nom_fichier = f"{chemin_xml.stem}_{numero}.jpg"
            cv2.imwrite(str(dossier_classe / nom_fichier), vignette)

            compteurs[nom_split][avion["classe"]] += 1

        if i % 500 == 0:
            print(f"  [{nom_split}] {i}/{len(liste_xml)} images traitées")


# ----------------------------------------------------------------------
# 5. Bilan
# ----------------------------------------------------------------------
print(f"\nVignettes ignorées (boîtes invalides) : {ignorees}")

toutes_classes = sorted(
    {c for compteur in compteurs.values() for c in compteur},
    key=lambda c: (len(c), c),                     # A1, A2, ..., A10, A11...
)
print(f"\n{'Classe':<8}{'train':>8}{'val':>8}{'test':>8}")
for classe in toutes_classes:
    print(f"{classe:<8}"
          f"{compteurs['train'][classe]:>8}"
          f"{compteurs['val'][classe]:>8}"
          f"{compteurs['test'][classe]:>8}")

total = sum(sum(c.values()) for c in compteurs.values())
print(f"\nTotal : {total} vignettes enregistrées dans {SORTIE}")
