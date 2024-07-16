# Résolution de grille de sudoku à partir d'une image avec un CNN


## Étapes pour résoudre un Sudoku à partir d'une image

### 1. Prétraitement de l'image
- Conversion de l'image en niveaux de gris pour simplifier l'analyse.
- Application de certain filtre puis binarisation de l'image

### 2. Localisation de la grille de Sudoku
- Détection des contours à l'aide de techniques de détection de contours.
- Localisation et extraction de la grille exacte du puzzle Sudoku.

### 3. Extraction des cellules
- Division de la grille détectée en ses 81 cellules individuelles (9x9).
- Isolation de chaque cellule pour un traitement ultérieur.

### 4. Reconnaissance des chiffres avec un CNN
- Préparation des cellules pour l'entrée du CNN (redimensionnement, normalisation).
- Utilisation d'un CNN pour reconnaître les chiffres de 1 à 9 sur chaque cellule.

### 5. Interprétation des résultats du CNN
- Conversion des prédictions du CNN en un format de grille numérique.

### 6. Résolution du Sudoku
- Implémentation d'un CNN de résolution de grille de sudoku avec Tensorflow et Keras.

## comment ecexuter le programme

les datasets sont trop volimuneux je n'ai pas pu les mettre sur github, voici les liens:
dataset pour modéle de résolution : https://www.kaggle.com/datasets/rohanrao/sudoku
dataset pour modéle de reconnaisance de chiffres: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

avant d'executer le main.py il faut d'abord executer reconnaisance2/index.py et resolution/app.py




