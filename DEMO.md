# Démo de la soutenance : aide-mémoire

## La veille (dimanche soir)
1. Choisir 2 images de démonstration avec plusieurs avions (dans yolo_dataset\images\test)
   et les copier sur le Bureau : démo_1.jpg, démo_2.jpg.
2. Faire une capture d'écran du résultat de l'appli sur chacune (plan B si ça plante).
3. Charger l'ordinateur et brancher le chargeur pour lundi.

## Avant d'entrer en salle (10 min avant)
1. Ouvrir VS Code dans le dossier "Dossier travail" (le terminal doit afficher (.venv)).
2. Dans le terminal :   python -m streamlit run app.py
3. Attendre l'ouverture de la page http://localhost:8501 dans le navigateur.
4. Tester UNE image pour "chauffer" le modèle (le 1er calcul est plus lent), puis laisser la page ouverte.
5. Fermer les autres programmes, couper les notifications.

## Pendant la démo (2 min)
1. Dire : "Je charge une image jamais vue par le modèle."
2. Glisser démo_1.jpg dans la zone de dépôt.
3. Montrer : les boîtes (YOLO), les classes A1 à A20 (ResNet), le tableau.
4. Bouger le curseur de seuil : "plus il monte, moins il y a de fausses alertes, mais on peut rater des avions".
5. Si ça plante : ouvrir la capture d'écran du plan B et expliquer la même chose.

## Si l'appli ne démarre pas
- Vérifier que (.venv) apparaît dans le terminal ; sinon : .venv\Scripts\activate
- Relancer : python -m streamlit run app.py
- Arrêter l'appli : Ctrl+C dans le terminal.
