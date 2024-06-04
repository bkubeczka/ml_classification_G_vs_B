# %%
# ****************************
# Choix du meilleur ajusté sur la base de leur score moyen par rééchantillonnage
# ****************************

import os
import logging
from joblib import load, dump
import logging
import toolbox_logging as log
import toolbox_constantes as cst

from numpy import mean 
from sklearn.model_selection import cross_val_score


#%%

logging_module_name = "004_job_best_model"
logging_level = logging.INFO

logger = log.getLogger(logging_module_name, logging_level)

log.print_phase(logger, "004. CHOIX du MEILLEUR MODELE pour le jeu de données JOB (job_best_model)")

#%%


#%% 

# =================
# LISTE des MODELES pour le jeu de données JOB
# =================

log.print_section(logger, "Chargement des modèles pour le jeu de données JOB")

# Chemin du répertoire contenant les fichiers
repertoire_models = f"{cst.MODELS_DIR}"

# Dictionnaire : nom de fichier => modèle sérialisé 
models = {}

# Parcours des fichiers <003_fitted_xxx.joblib> du répertoire
for nom_fichier in os.listdir(repertoire_models):
    
    # Vérification du préfixe et de l'extension .joblib
    if nom_fichier.startswith('003_job_fitted') and nom_fichier.endswith('.joblib'):
        
        logger.info(f'{nom_fichier}')
        
        # Deserialisation du modèle
        # -------------------------
        model = load(repertoire_models + "/" + nom_fichier)
        
        # Stockage dans le dictionnaire
        # --------------------------
        models[nom_fichier] = model
        
# Affichage des noms des fichiers trouvés
logger.debug(f'{models}\n')
logger.info('\n')

#%%

# ==========================
# COLLECTE des SCORES MOYENS
# ==========================

log.print_section(logger, "Collecte / calcul des scores moyens des modèles sur cross-validation")


# Chargement des objets sérialisé 
# (utile pour la validation croisée si besoin)
# -------------------------------

X_train = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_X_TRAIN}")
y_train = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_Y_TRAIN}")
resample = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_RESAMPLE}")


# Collecte du score moyen des modèles
# -----------------------------------
#
# 2 cas à considérer
#
# . le modèle a été construit par GSCV avec une évaluation par rééchantillonnage
#   => on lit le score moyen stocké dans le modèle ajusté
#
# . le modèle n'a pas été optimisé par GSCV (logistic regression / Tree avec élagage)
#   => on évalue la performance du modèle ajusté par cross_val_score
#   => on calcule le score moyen à l'issu
#
scores = {} # Dictionnaire nom de fichier => score du modèle 
for model_name, model in models.items():
    
    score = 0
    
    try:
        # collecte de l'attribut mean_test_score (s'il existe)
        # --------------------------------------
        score = model.cv_results_["mean_test_score"][model.best_index_]

        logger.info(f'{model_name}:')
        logger.info(f'{score}')
        
    except:
        
        # s'il n'existe pas 
        # . évaluation sur le jeu Train par la technique commmune de rééchantillonnage
        # . Calcul du score moyen
        # ---------------------------------------------------------

        model_scores = cross_val_score ( 
            X=X_train,
            y=y_train,
            estimator=model, 
            cv=resample, 
            scoring="accuracy", 
            n_jobs=-1)

        score = mean(model_scores)
        
        logger.info(f'{model_name}:')
        logger.info(f'{model_scores} - score moyen = {score}')

        
    # Ajouter le score au dictionnaire
    # --------------------------------
    scores[model_name] = score

logger.info("")    
logger.debug(f'{scores}')
logger.debug('')

#%%

# ========================
# CHOIX du MEILLEUR MODELE
# ========================

# Le modèle choisi est celui qui a obtenu le meilleur score moyen par cross-validation
# ---------------------------------------------------------------
best_model_name = max(scores, key=lambda k: scores[k])

logger.info(f'JOB Best model : {best_model_name} (score={scores[best_model_name]})')
logger.info('\n')

# Sérialisation du nom de fichier pour étape suivante (prédiction)
# -------------------------------
dump(best_model_name, f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_BEST_MODEL}")
