import os
import statistics
import matplotlib.pyplot as plt
from PIL import Image

dossier_images = r"C:\Users\louis\Documents\Cours\IA\military-aircraft-recognition-dataset.train\student-dataset 2\images"

largeurs = []
hauteurs = []

# 1. Lecture des images et extraction des dimensions
if not os.path.exists(dossier_images):
    print("ERREUR : Le dossier des images est introuvable. Vérifie le chemin.")
else:
    fichiers = [f for f in os.listdir(dossier_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Analyse de {len(fichiers)} images en cours...")

    for fichier in fichiers:
        chemin_complet = os.path.join(dossier_images, fichier)
        try:
            with Image.open(chemin_complet) as img:
                largeur, hauteur = img.size
                largeurs.append(largeur)
                hauteurs.append(hauteur)
        except Exception as e:
            print(f"Erreur de lecture sur l'image {fichier} : {e}")

# 2. Calcul des statistiques et génération du graphique
if largeurs and hauteurs:
    moy_l = statistics.mean(largeurs)
    med_l = statistics.median(largeurs)
    ecart_l = statistics.stdev(largeurs) if len(largeurs) > 1 else 0
    
    moy_h = statistics.mean(hauteurs)
    med_h = statistics.median(hauteurs)
    ecart_h = statistics.stdev(hauteurs) if len(hauteurs) > 1 else 0

    plt.figure(figsize=(10, 6))
    plt.scatter(largeurs, hauteurs, alpha=0.5, color='coral', edgecolor='black', label='Images du dataset')
    plt.scatter(moy_l, moy_h, color='red', marker='X', s=150, label='Moyenne globale', zorder=5)

    # Préparation et ajout du bloc de texte des statistiques en haut à droite
    texte_stats = (
        "Statistiques (Largeur | Hauteur)\n"
        "-------------------------------------\n"
        f"Moyenne    : {moy_l:.0f} px | {moy_h:.0f} px\n"
        f"Médiane    : {med_l:.0f} px | {med_h:.0f} px\n"
        f"Écart-type : {ecart_l:.0f} px | {ecart_h:.0f} px"
    )

    plt.text(0.95, 0.95, texte_stats, 
             transform=plt.gca().transAxes, 
             fontsize=10, 
             verticalalignment='top', 
             horizontalalignment='right', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))

    # Personnalisation du graphique
    plt.title("Distribution des dimensions des images d'entraînement", fontsize=14, fontweight='bold')
    plt.xlabel("Largeur (pixels)", fontsize=12)
    plt.ylabel("Hauteur (pixels)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Positionnement de la légende en bas à droite pour ne pas cacher le texte
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Sauvegarde
    chemin_sauvegarde = r"C:\Users\louis\Documents\Cours\IA\stats_dimensions_nuage.png"
    plt.savefig(chemin_sauvegarde, dpi=300)
    print(f"\nLe graphique a été sauvegardé avec succès ici : {chemin_sauvegarde}")

    # Affichage à l'écran
    plt.show()
else:
    print("Aucune donnée à analyser pour générer les statistiques.")