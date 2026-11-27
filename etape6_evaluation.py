"""
Étape 6 : évaluation de bout en bout (YOLO + ResNet) sur les images de TEST.

On compare ce que le pipeline prédit avec les vraies annotations XML :
  - un avion prédit est "trouvé" s'il recouvre un vrai avion avec IoU >= 0.5 ;
  - il est "bien identifié" s'il est trouvé ET que sa classe (A1...A20) est la bonne.

Chiffres produits (à mettre dans le rapport et l'oral) :
  - précision / rappel de la DÉTECTION
  - précision de la CLASSIFICATION sur les avions trouvés
  - précision / rappel de BOUT EN BOUT (trouvé ET bien identifié)

Sorties dans le dossier "evaluation/" :
  - bilan.txt, details_par_image.csv, matrice_confusion_bout_en_bout.png

Prérequis : etape5_pipeline.py dans le même dossier + ses fichiers
(sauvegardes/best.pt, sauvegardes/meilleur_modele.pth).
"""

from pathlib import Path
import csv
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")           # pas de fenêtre : on enregistre juste l'image
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from ultralytics import YOLO

import etape5_pipeline as pipe   # réutilise les fonctions de l'étape 5

# ----------------------------------------------------------------------
# 1. Paramètres
# ----------------------------------------------------------------------
# Dossier des ANNOTATIONS XML d'origine (même chemin qu'aux étapes 2 et 4a)
DOSSIER_ANNOTATIONS = Path(
    r"C:\Users\louis\Documents\Cours\IA\military-aircraft-recognition-dataset.train"
    r"\student-dataset 2\annotations"
)
# Images de test (le découpage créé à l'étape 4a, identique à l'étape 2)
DOSSIER_IMAGES_TEST = pipe.DOSSIER_SCRIPT / "yolo_dataset" / "images" / "test"
DOSSIER_SORTIE = pipe.DOSSIER_SCRIPT / "evaluation"

SEUIL_IOU = 0.5          # recouvrement minimal pour dire "c'est le même avion"
NB_IMAGES_MAX = None     # None = toutes les images de test


# ----------------------------------------------------------------------
# 2. Outils
# ----------------------------------------------------------------------
def lire_verite(chemin_xml):
    """Vraies boîtes et classes d'un XML : liste de (x1, y1, x2, y2, classe)."""
    racine = ET.parse(chemin_xml).getroot()
    avions = []
    for objet in racine.findall("object"):
        b = objet.find("bndbox")
        avions.append((
            float(b.find("xmin").text), float(b.find("ymin").text),
            float(b.find("xmax").text), float(b.find("ymax").text),
            objet.find("name").text,
        ))
    return avions


def iou(a, b):
    """IoU = surface commune / surface totale de deux rectangles (0 à 1)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    commun = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aire_a = (a[2] - a[0]) * (a[3] - a[1])
    aire_b = (b[2] - b[0]) * (b[3] - b[1])
    total = aire_a + aire_b - commun
    return commun / total if total > 0 else 0.0


def associer(predictions, verite, seuil=SEUIL_IOU):
    """Associe chaque prédiction à au plus un vrai avion (les plus confiantes d'abord).

    Renvoie : liste de (indice_prediction, indice_verite), prédictions sans vrai
    avion, vrais avions non trouvés.
    """
    ordre = sorted(range(len(predictions)),
                   key=lambda i: -predictions[i]["confiance_detection"])
    deja_pris = set()
    paires = []
    for i in ordre:
        p = predictions[i]
        boite_p = (p["x1"], p["y1"], p["x2"], p["y2"])
        meilleur, meilleur_iou = None, seuil
        for j, v in enumerate(verite):
            if j in deja_pris:
                continue
            valeur = iou(boite_p, v[:4])
            if valeur >= meilleur_iou:
                meilleur, meilleur_iou = j, valeur
        if meilleur is not None:
            deja_pris.add(meilleur)
            paires.append((i, meilleur))
    associees_p = {i for i, _ in paires}
    fausses_alertes = [i for i in range(len(predictions)) if i not in associees_p]
    manques = [j for j in range(len(verite)) if j not in deja_pris]
    return paires, fausses_alertes, manques


def rapport(cible, valeur):
    return f"{valeur:.1%}" if cible else "n/a"


# ----------------------------------------------------------------------
# 3. Programme principal
# ----------------------------------------------------------------------
def main():
    for chemin in (pipe.CHEMIN_YOLO, pipe.CHEMIN_RESNET):
        if not chemin.exists():
            raise SystemExit(f"Fichier introuvable : {chemin}")
    if not DOSSIER_IMAGES_TEST.exists():
        raise SystemExit(f"Dossier introuvable : {DOSSIER_IMAGES_TEST}")
    if not DOSSIER_ANNOTATIONS.exists():
        raise SystemExit(f"Dossier d'annotations introuvable : {DOSSIER_ANNOTATIONS}\n"
                         "Corrige DOSSIER_ANNOTATIONS en haut du script.")

    yolo = YOLO(str(pipe.CHEMIN_YOLO))
    resnet, classes = pipe.charger_resnet(pipe.CHEMIN_RESNET)

    images = sorted(p for p in DOSSIER_IMAGES_TEST.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if NB_IMAGES_MAX is not None:
        images = images[:NB_IMAGES_MAX]
    print(f"{len(images)} images de test à évaluer (calcul sur {pipe.device})")

    total_vrais = total_predits = total_trouves = total_bien_identifies = 0
    vraies_classes, classes_predites = [], []
    lignes = []

    for numero, chemin_image in enumerate(images, start=1):
        chemin_xml = DOSSIER_ANNOTATIONS / f"{chemin_image.stem}.xml"
        if not chemin_xml.exists():
            print(f"  XML introuvable pour {chemin_image.name}, ignorée")
            continue
        verite = lire_verite(chemin_xml)
        predictions = pipe.analyser_image(chemin_image, yolo, resnet, classes)
        paires, fausses, manques = associer(predictions, verite)

        bien = 0
        for i, j in paires:
            vraie, predite = verite[j][4], predictions[i]["classe"]
            vraies_classes.append(vraie)
            classes_predites.append(predite)
            bien += int(vraie == predite)

        total_vrais += len(verite)
        total_predits += len(predictions)
        total_trouves += len(paires)
        total_bien_identifies += bien
        lignes.append({
            "image": chemin_image.name, "vrais_avions": len(verite),
            "avions_predits": len(predictions), "trouves": len(paires),
            "bien_identifies": bien, "fausses_alertes": len(fausses),
            "manques": len(manques),
        })
        if numero % 25 == 0:
            print(f"  {numero}/{len(images)} images traitées")

    # ------------------------------------------------------------------
    # Chiffres finaux
    # ------------------------------------------------------------------
    det_precision = total_trouves / total_predits if total_predits else 0.0
    det_rappel = total_trouves / total_vrais if total_vrais else 0.0
    classif = total_bien_identifies / total_trouves if total_trouves else 0.0
    bout_precision = total_bien_identifies / total_predits if total_predits else 0.0
    bout_rappel = total_bien_identifies / total_vrais if total_vrais else 0.0

    texte = (
        f"Évaluation de bout en bout sur {len(lignes)} images de test\n"
        f"(IoU >= {SEUIL_IOU}, seuil de détection YOLO = {pipe.SEUIL_DETECTION})\n\n"
        f"Vrais avions : {total_vrais} | avions prédits : {total_predits} | "
        f"trouvés : {total_trouves} | trouvés ET bien identifiés : {total_bien_identifies}\n\n"
        f"DÉTECTION    précision {det_precision:.1%} | rappel {det_rappel:.1%}\n"
        f"CLASSIFICATION (sur les avions trouvés) : {classif:.1%}\n"
        f"BOUT EN BOUT précision {bout_precision:.1%} | rappel {bout_rappel:.1%}\n"
    )
    print("\n" + texte)

    DOSSIER_SORTIE.mkdir(exist_ok=True)
    (DOSSIER_SORTIE / "bilan.txt").write_text(texte, encoding="utf-8")
    with open(DOSSIER_SORTIE / "details_par_image.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        ecrivain = csv.DictWriter(f, fieldnames=list(lignes[0].keys()), delimiter=";")
        ecrivain.writeheader()
        ecrivain.writerows(lignes)

    if vraies_classes:
        labels = sorted(set(classes), key=lambda c: (len(c), c))
        figure, axe = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay.from_predictions(
            vraies_classes, classes_predites, labels=labels, display_labels=labels,
            ax=axe, xticks_rotation=90, colorbar=False)
        axe.set_title("Matrice de confusion de bout en bout (avions trouvés par YOLO)")
        plt.tight_layout()
        plt.savefig(DOSSIER_SORTIE / "matrice_confusion_bout_en_bout.png", dpi=100)
    print(f"Résultats enregistrés dans : {DOSSIER_SORTIE}")


if __name__ == "__main__":
    main()
