"""
Étape 7 : interface graphique (Streamlit) du détecteur d'avions.

Lancer (dans le .venv) :
    python -m pip install streamlit
    python -m streamlit run app.py

Fonctionnement : on dépose une image satellite, YOLO trouve les avions,
le ResNet identifie leur modèle (A1 à A20), et le résultat s'affiche
avec un tableau téléchargeable.
"""

import tempfile
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

import etape5_pipeline as pipe

st.set_page_config(page_title="Détection d'avions", layout="wide")
st.title("Détection et identification d'avions sur images satellites")
st.caption("YOLO localise les avions, un ResNet18 (Transfer Learning) identifie leur modèle.")


@st.cache_resource(show_spinner="Chargement des modèles...")
def charger_modeles():
    """Charge YOLO et le ResNet une seule fois (ils restent en mémoire)."""
    yolo = YOLO(str(pipe.CHEMIN_YOLO))
    resnet, classes = pipe.charger_resnet(pipe.CHEMIN_RESNET)
    return yolo, resnet, classes


yolo, resnet, classes = charger_modeles()

with st.sidebar:
    st.header("Réglages")
    seuil = st.slider(
        "Seuil de confiance de la détection", 0.05, 0.95, float(pipe.SEUIL_DETECTION), 0.05,
        help="Plus il est haut, moins il y a de fausses alertes, mais plus on risque de rater des avions.",
    )
    st.write(f"Calcul sur : **{pipe.device}**")

fichier = st.file_uploader("Dépose une image satellite", type=["jpg", "jpeg", "png"])

if fichier is not None:
    pipe.SEUIL_DETECTION = seuil          # lu par analyser_image à chaque appel
    with tempfile.TemporaryDirectory() as dossier_tmp:
        chemin = Path(dossier_tmp) / fichier.name
        chemin.write_bytes(fichier.getbuffer())

        with st.spinner("Analyse en cours..."):
            detections = pipe.analyser_image(chemin, yolo, resnet, classes)
        sortie = Path(dossier_tmp) / ("annotee_" + fichier.name)
        pipe.dessiner(chemin, detections, sortie)
        image_annotee = cv2.cvtColor(cv2.imread(str(sortie)), cv2.COLOR_BGR2RGB)

    colonne_image, colonne_tableau = st.columns([3, 2])
    with colonne_image:
        st.image(image_annotee, caption=f"{len(detections)} avion(s) détecté(s)",
                 use_container_width=True)
    with colonne_tableau:
        if detections:
            lignes = [{
                "Modèle": d["classe"],
                "Confiance modèle": f'{d["confiance_classification"]:.0%}',
                "Confiance détection": f'{d["confiance_detection"]:.0%}',
                "Position (x1, y1, x2, y2)": f'{d["x1"]}, {d["y1"]}, {d["x2"]}, {d["y2"]}',
            } for d in detections]
            st.dataframe(lignes, use_container_width=True)
            repartition = {}
            for d in detections:
                repartition[d["classe"]] = repartition.get(d["classe"], 0) + 1
            st.write("Répartition :", ", ".join(f"{c} × {n}" for c, n in sorted(repartition.items())))
        else:
            st.info("Aucun avion trouvé avec ce seuil. Essaie de le baisser.")
else:
    st.info("Dépose une image pour lancer l'analyse.")
