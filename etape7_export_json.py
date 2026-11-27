"""
Étape 7 : fabriquer le fichier de prédictions au format de la page d'évaluation du prof.

Format attendu (comme "example-military-dataset-annotations.json") :
{
  "1.jpg": [ {"class": "A2", "coordinates": {"xmin": 485, "ymin": 427, "xmax": 554, "ymax": 500}}, ... ],
  "10.jpg": [ ... ]
}

Utilisation (dans le terminal, .venv activé, dossier "Dossier travail") :
    python etape7_export_json.py "<dossier des images d'évaluation>"
Le fichier predictions.json est créé à côté du script.
"""

from pathlib import Path
import json
import sys

from ultralytics import YOLO

import etape5_pipeline as pipe   # réutilise YOLO + ResNet de l'étape 5

DOSSIER_IMAGES = Path(
    r"C:\Users\louis\Documents\Cours\IA\military-aircraft-recognition-dataset.eval (1)\eval-dataset\images"
)
if len(sys.argv) >= 2:
    DOSSIER_IMAGES = Path(sys.argv[1])
FICHIER_SORTIE = pipe.DOSSIER_SCRIPT / "predictions.json"


def main():
    if not DOSSIER_IMAGES.exists():
        raise SystemExit(f"Dossier introuvable : {DOSSIER_IMAGES}")
    yolo = YOLO(str(pipe.CHEMIN_YOLO))
    resnet, classes = pipe.charger_resnet(pipe.CHEMIN_RESNET)

    images = sorted(p for p in DOSSIER_IMAGES.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"{len(images)} images à traiter (calcul sur {pipe.device})")

    resultat = {}
    for numero, chemin in enumerate(images, start=1):
        avions = pipe.analyser_image(chemin, yolo, resnet, classes)
        resultat[chemin.name] = [
            {"class": a["classe"],
             "coordinates": {"xmin": int(a["x1"]), "ymin": int(a["y1"]),
                             "xmax": int(a["x2"]), "ymax": int(a["y2"])}}
            for a in avions
        ]
        if numero % 100 == 0:
            print(f"  {numero}/{len(images)}")

    FICHIER_SORTIE.write_text(json.dumps(resultat, indent=2), encoding="utf-8")
    total = sum(len(v) for v in resultat.values())
    print(f"{total} avions écrits pour {len(resultat)} images dans : {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()
