"""
Étape 3 : entraîner un classifieur d'avions (ResNet18 pré-entraîné).

Ce script :
  1. charge les vignettes de crops/train, crops/val et crops/test ;
  2. charge un ResNet18 pré-entraîné sur ImageNet (Transfer Learning) ;
  3. remplace sa dernière couche pour qu'elle prédise TES classes ;
  4. entraîne sur train, surveille val, garde le meilleur modèle ;
  5. évalue le meilleur modèle sur test (une seule fois, à la fin).

Installation (voir les explications dans la conversation) :
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    python -m pip install scikit-learn matplotlib
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Paramètres
# ----------------------------------------------------------------------
DOSSIER_CROPS = Path(__file__).parent / "crops"   # le dossier créé à l'étape 2
NB_EPOCHS = 10          # nombre de passages complets sur les données d'entraînement
TAILLE_LOT = 64         # nombre de vignettes traitées en même temps
TAUX_APPRENTISSAGE = 1e-4
TAILLE_IMAGE = 224      # taille attendue par ResNet

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Entraînement sur : {device}")
    if device == "cpu":
        print("ATTENTION : le GPU n'est pas utilisé, ce sera très lent. "
              "PyTorch a probablement été installé en version CPU.")

    # ------------------------------------------------------------------
    # 2. Préparer les images
    #    - train : on ajoute de la Data Augmentation (retournements, rotation)
    #    - val / test : aucune modification aléatoire, juste le redimensionnement
    # ------------------------------------------------------------------
    normalisation = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # moyennes et écarts-types d'ImageNet,
        std=[0.229, 0.224, 0.225],    # attendus par le ResNet pré-entraîné
    )

    transfo_train = transforms.Compose([
        transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),     # un avion peut pointer dans toutes les directions
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalisation,
    ])
    transfo_eval = transforms.Compose([
        transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)),
        transforms.ToTensor(),
        normalisation,
    ])

    # ImageFolder : un sous-dossier = une classe (A1, A2...)
    data_train = datasets.ImageFolder(DOSSIER_CROPS / "train", transfo_train)
    data_val = datasets.ImageFolder(DOSSIER_CROPS / "val", transfo_eval)
    data_test = datasets.ImageFolder(DOSSIER_CROPS / "test", transfo_eval)

    classes = data_train.classes
    print(f"{len(classes)} classes : {classes}")
    print(f"Vignettes : {len(data_train)} train, {len(data_val)} val, {len(data_test)} test")

    # Sécurité : val et test doivent avoir les mêmes classes que train,
    # sinon les numéros de classes ne correspondraient pas.
    if data_val.classes != classes or data_test.classes != classes:
        raise SystemExit(
            "Les dossiers train/val/test n'ont pas les mêmes classes "
            "(une classe est absente de val ou test). Dis-le-moi, on adaptera."
        )

    # DataLoader : fournit les vignettes par lots, dans un ordre mélangé pour train
    loader_train = DataLoader(data_train, batch_size=TAILLE_LOT, shuffle=True, num_workers=0)
    loader_val = DataLoader(data_val, batch_size=TAILLE_LOT, num_workers=0)
    loader_test = DataLoader(data_test, batch_size=TAILLE_LOT, num_workers=0)

    # ------------------------------------------------------------------
    # 3. Le modèle : ResNet18 pré-entraîné + nouvelle dernière couche
    # ------------------------------------------------------------------
    modele = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    modele.fc = nn.Linear(modele.fc.in_features, len(classes))   # une sortie par classe
    modele = modele.to(device)

    fonction_perte = nn.CrossEntropyLoss()     # mesure l'erreur de classification
    optimiseur = torch.optim.Adam(modele.parameters(), lr=TAUX_APPRENTISSAGE)

    # ------------------------------------------------------------------
    # 4. Fonctions pour faire passer un lot de données dans le modèle
    # ------------------------------------------------------------------
    def entrainer_une_epoch():
        modele.train()                          # mode entraînement
        perte_totale, bonnes_reponses = 0.0, 0
        for images, etiquettes in loader_train:
            images, etiquettes = images.to(device), etiquettes.to(device)

            optimiseur.zero_grad()              # remet les gradients à zéro
            sorties = modele(images)            # prédiction
            perte = fonction_perte(sorties, etiquettes)
            perte.backward()                    # calcule les gradients
            optimiseur.step()                   # descente de gradient : met à jour le modèle

            perte_totale += perte.item() * images.size(0)
            bonnes_reponses += (sorties.argmax(1) == etiquettes).sum().item()
        n = len(data_train)
        return perte_totale / n, bonnes_reponses / n

    def evaluer(loader, dataset):
        modele.eval()                           # mode évaluation
        perte_totale, bonnes_reponses = 0.0, 0
        toutes_predictions, toutes_etiquettes = [], []
        with torch.no_grad():                   # pas d'apprentissage ici
            for images, etiquettes in loader:
                images, etiquettes = images.to(device), etiquettes.to(device)
                sorties = modele(images)
                perte_totale += fonction_perte(sorties, etiquettes).item() * images.size(0)
                predictions = sorties.argmax(1)
                bonnes_reponses += (predictions == etiquettes).sum().item()
                toutes_predictions += predictions.cpu().tolist()
                toutes_etiquettes += etiquettes.cpu().tolist()
        n = len(dataset)
        return perte_totale / n, bonnes_reponses / n, toutes_etiquettes, toutes_predictions

    # ------------------------------------------------------------------
    # 5. Boucle d'entraînement
    # ------------------------------------------------------------------
    meilleure_precision_val = 0.0
    for epoch in range(1, NB_EPOCHS + 1):
        perte_train, prec_train = entrainer_une_epoch()
        perte_val, prec_val, _, _ = evaluer(loader_val, data_val)

        print(f"Epoch {epoch:>2}/{NB_EPOCHS} | "
              f"train : perte {perte_train:.3f}, précision {prec_train:.1%} | "
              f"val : perte {perte_val:.3f}, précision {prec_val:.1%}")

        # On sauvegarde le modèle seulement s'il est meilleur sur val
        if prec_val > meilleure_precision_val:
            meilleure_precision_val = prec_val
            torch.save({"poids": modele.state_dict(), "classes": classes},
                       "meilleur_modele.pth")
            print("   -> meilleur modèle sauvegardé")

    # ------------------------------------------------------------------
    # 6. Examen final sur test avec le meilleur modèle
    # ------------------------------------------------------------------
    sauvegarde = torch.load("meilleur_modele.pth", map_location=device)
    modele.load_state_dict(sauvegarde["poids"])
    _, prec_test, etiquettes, predictions = evaluer(loader_test, data_test)

    print(f"\nPrécision finale sur test : {prec_test:.1%}\n")
    toutes = list(range(len(classes)))
    print(classification_report(etiquettes, predictions, labels=toutes,
                                target_names=classes, zero_division=0))

    figure, axe = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        etiquettes, predictions, labels=toutes, display_labels=classes,
        ax=axe, xticks_rotation=90, colorbar=False,
    )
    plt.tight_layout()
    plt.savefig("matrice_confusion.png", dpi=100)
    print("Matrice de confusion sauvegardée dans matrice_confusion.png")
    plt.show()


# Cette ligne est obligatoire sous Windows quand on utilise num_workers > 0
if __name__ == "__main__":
    main()
