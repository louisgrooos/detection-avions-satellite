# À faire par toi (le reste est préparé)

Toutes les commandes se lancent dans le terminal de VS Code, dossier "Dossier travail",
avec (.venv) visible au début de la ligne.

## 1. Installer (une fois)
python -m pip install ultralytics streamlit pillow

## 2. Test rapide du pipeline (10 images de test)
python etape5_pipeline.py
-> ouvre le dossier "resultats" : les images doivent montrer des rectangles + des étiquettes A1..A20.
-> si erreur : copie-la-moi en entier.

## 3. Chiffres de bout en bout (pour le rapport et l'oral)
python etape6_evaluation.py
-> envoie-moi le contenu de evaluation\bilan.txt

## 4. Interface
python -m streamlit run app.py
-> dépose une image, vérifie que ça affiche les avions. (Ctrl+C pour arrêter)

## 5. Corpus d'évaluation du prof (UNE SEULE FOIS, à la fin, sans rien régler après)
python etape5_pipeline.py "C:\Users\louis\Documents\Cours\IA\military-aircraft-recognition-dataset.eval (1)\eval-dataset\images" "C:\Users\louis\Documents\Cours\IA\Dossier travail\corpus_eval_annote"
-> zippe le dossier corpus_eval_annote, mets-le sur Google Drive (lien partageable) : c'est le "lien vers le corpus annoté" du rapport.

## 6. Git
Crée un dépôt GitHub (privé ou public selon ce que demande le prof), puis dans le terminal :
git init
git add .
git commit -m "Projet detection d'avions"
(le .gitignore exclut déjà les gros dossiers ; best.pt et meilleur_modele.pth sont inclus)
-> envoie-moi le lien du dépôt.

## 7. Docker (optionnel, si Docker Desktop est installé)
docker build -t avions .
docker run -p 8501:8501 avions
