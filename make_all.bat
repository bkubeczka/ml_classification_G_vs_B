@echo off

:: Installation des packages requis pour l'exécution du code
call pip install -r ./requirements.txt

:: Jeu de données
call python 001_make_dataset.py

:: Apprentissage
call python 002_preprocessing_main.py
call python 003_main_model_gradient_boosting.py
call python 003_main_model_logistic_regression.py

call python 002_preprocessing_job.py
call python 003_job_model_gradient_boosting.py
call python 003_job_model_logistic_regression.py

:: Performance et Prédictions
call python 004_main_best_model.py
call python 004_job_best_model.py
call python 005_predictions.py

:: Rapport
call quarto render "ml_projet3_kubeczka.qmd" --to pdf

