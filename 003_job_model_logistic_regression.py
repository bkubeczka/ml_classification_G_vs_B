# %% LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from joblib import dump, load
import logging
import toolbox_logging as log
import toolbox_constantes as cst

#%% LOGGING

logging_module_name = "003_job_model_logistic_regression"
logging_level = logging.DEBUG

logger = log.getLogger(logging_module_name, logging_level)

log.print_phase(logger, "003. MODELE REGRESSION LOGISTIQUE sur jeu de données JOB")

# %%
# =======================================
# Désérialisation des données de construction des modèles
# =======================================

log.print_section(logger, "Désérialisation des objets", ln=True)

# jeux de données train / test pour la construction et l'évaluation des modèles
logger.debug(f"{cst.JOBLIB_002_JOB_X_TRAIN}")
X_train = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_X_TRAIN}")  # jeu d'entraînement
logger.debug(f"{cst.JOBLIB_002_JOB_Y_TRAIN}")
y_train = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_Y_TRAIN}")  # jeu d'entraînement

# pipeline commun
logger.debug(f"{cst.JOBLIB_002_JOB_COMMON_PIPELINE}")
pipeline_common = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_COMMON_PIPELINE}") # pipeline commun à tous les modèles

# Stratégie d'évaluation des modèles candidats
logger.debug(f"{cst.JOBLIB_002_JOB_RESAMPLE}")
resample = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_RESAMPLE}") # stratégie d'évaluation des modèles
logger.info("")

# %%
# ================================
# MODELE : Régression Logistique avec optimisation par GSCV
# ================================
#
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import GridSearchCV


# --------------------
# Création du modèle Logistic Regression Classifier
# --------------------
#
log.print_section(logger, "Création LogisticRegression")
model_LR = LogisticRegression (max_iter=5000, random_state=50, n_jobs=-1)
logger.info(model_LR)
logger.info("")
                                      
# ----------------------------------                                      
# Création du pipeline de traitement des données
# ----------------------------------
log.print_section(logger, "Création pipeline de traitement de données")

# Copie du pipeline commun
# ------------------------
logger.info("...copie du pipeline commun de création des features (cf. 002_preprocessing)")
pipeline_LR = deepcopy(pipeline_common)

# pipeline spécifique pour le modèle
# ----------------------------------
# . Suppression des colonnes à forte corrélation
# . toute variable catégorielle : OHE (contrainte sklearn) avec suppression de la 1ère colonne (inversibilité de la matrice) / ignorer les modalités inconnues
# . toute variable numérique : mise à l'échelle avec StandardScaler
#
logger.info("...ajout du preprocessing dédié au modèle LR")
logger.info("numérique    : mise à l'échelle")
logger.info("categorielle : OHE")

columns_to_drop = ['REGION_', 'CSP_N1_', 'ACTIVITY_L1_', 'HOUSEHOLD_L1_',
                   'WD_N1_', 'WD_N2_',
                  ] # Suppression de colonnes pour cause de fortes corrélations

logger.info("drop : {c}" for c in columns_to_drop)

preprocessing_LR = ColumnTransformer(
    transformers=[
        ('drop', 'drop', columns_to_drop),
        ('numeric', StandardScaler(), make_column_selector(dtype_include='number')), # mise à l'échelle pour aligner l'échelle des valeurs numériques
        ('category', OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore'), make_column_selector(dtype_include=['category','bool'])),
    ], 
    remainder="passthrough", 
    verbose_feature_names_out=False
)

pipeline_LR.steps.append(("preprocessing", preprocessing_LR))       

pipeline_LR.steps.append(("estimator", model_LR))

logger.info("")
logger.info(f'------- Pipeline -------')
logger.info(pipeline_LR)
logger.info('')   

# -------------------------
# Création de la grille de méta-paramètres
# -------------------------

# . penalty : "none" / "l2"
# . solver : "liblinear" / "sag" / "newton-cholesky"
# . C : coefficient de régularisation (>0) (1/alpha) : on utilise une grande valeur pour "désactiver" la régularisation

  
log.print_section(logger, "Création de la grille")
grid_LR = { 
            'estimator__penalty': ["l2"],
            'estimator__solver': ["sag", "lbfgs", "newton-cholesky"],
            'estimator__C': [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
}

logger.info(grid_LR)
logger.info("")

# --------------------
# Ajustement du modèle par GSCV
# --------------------
# . Données : pipeline feature + preprocessing LR
# . application des méta-paramètres de la grille définie
# . politique commune de mesure des performance (resampling K-Fold à 5 blocs)
# 

import warnings 
warnings.filterwarnings(action="once", category=UserWarning)

log.print_section(logger, "Création du GSCV")
GSCV_LR = GridSearchCV(
                  estimator=pipeline_LR, 
                  param_grid=grid_LR,
                  cv=resample,
                  scoring="accuracy", 
                  refit=True,
                  n_jobs=-1
)

logger.info("")
logger.debug(GSCV_LR)
logger.info("")

logger.info("... ajustement du GSCV")
optimised_LR = GSCV_LR.fit(X_train, y_train)
logger.info("")

# Meilleur modèle
# ---------------
log.print_section(logger, "RESULTATS : meilleur modèle")
logger.info(f'paramètres: {optimised_LR.best_params_}')
logger.info("")

# Scores par blocs / moyen
# ------------------------
cv_data = ['mean_test_score', 'std_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score','split4_test_score']
logger.info(f'scores by split:')
for cv in cv_data:
  logger.info(f'{cv:>20}: {optimised_LR.cv_results_[cv][optimised_LR.best_index_]:02f}')
logger.info("")

# Scores
# ------
logger.info(f'score sur job_X_train: {accuracy_score(y_train, optimised_LR.predict(X_train))}')
logger.info("")  


# ==============================
# SERIALISATION du MODELE AJUSTE
# ==============================

log.print_section(logger, "Sérialisation du modèle ajusté")

# Sauvegarde du modèle ajusté (pour rapport)
# ---------------------------
logger.info("003_job_fitted_logistic_regression.joblib")
dump(optimised_LR, f"{cst.MODELS_DIR}/003_job_fitted_logistic_regression.joblib")
logger.info("")
