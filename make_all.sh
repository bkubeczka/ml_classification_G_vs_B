#!/bin/bash

# Installation des packages requis pour l'exécution du code
pip3 install -r ./requirements.txt

# Jeu de données
python3 ./001_make_dataset.py

# Apprentissage
python3 ./002_preprocessing_main.py
python3 ./003_main_model_gradient_boosting.py
python3 ./003_main_model_logistic_regression.py

python3 ./002_preprocessing_job.py
python3 ./003_job_model_gradient_boosting.py
python3 ./003_job_model_logistic_regression.py

# Performance et Prédictions
python3 ./004_main_best_model.py
python3 ./004_job_best_model.py
python3 ./005_predictions.py

# Rapport
quarto render ./ml_projet3_kubeczka.qmd --to pdf
