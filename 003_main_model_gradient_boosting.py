# %% LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from joblib import dump, load
import logging
import toolbox_logging as log
import toolbox_constantes as cst

#%% LOGGING



logging_module_name = "003_main_model_gradient_boosting"
logging_level = logging.DEBUG

logger = log.getLogger(logging_module_name, logging_level)

log.print_phase(logger, "003. Jeu de données MAIN : MODELE GRADIENT BOOSTING")


# %%
# =======================================
# Désérialisation des données de construction des modèles
# =======================================

log.print_section(logger, "Désérialisation des objets")

# jeux de données train / test pour la construction et l'évaluation des modèles
logger.debug(f"{cst.JOBLIB_002_MAIN_X_TRAIN}")
X_train = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_X_TRAIN}")  # jeu d'entraînement
logger.debug(f"{cst.JOBLIB_002_MAIN_Y_TRAIN}")
y_train = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_Y_TRAIN}")  # jeu d'entraînement

# pipeline commun
logger.debug(f"{cst.JOBLIB_002_MAIN_COMMON_PIPELINE}")
pipeline_common = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_COMMON_PIPELINE}") # pipeline commun à tous les modèles

# Stratégie d'évaluation des modèles candidats
logger.debug(f"{cst.JOBLIB_002_MAIN_RESAMPLE}")
resample = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_RESAMPLE}") # stratégie d'évaluation des modèles
logger.info("")

# %%
# ================================
# MODELE : Gradient Boosting - optimisation par GSCV
# ================================
#
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import GridSearchCV

# ----------------
# Création du Modèle Gradient Boosting
# ----------------
#
# . n_estimators : Nombre d'arbres construits dans par l'algortihme
# . max_samples_split = 20 : 20 éléments minimum dans le noeud pour poursuivre la segmentation
# . max_samples_leaf = 6 : 6 éléments mini pour qu'une segmentation soit considérée
# . min_impurity_decrease : pas de pruning => on impose un gain d'impureté pour lever le surapprentissage
#
log.print_section(logger, "Création du GradientBoostingClassifier")
model_GB = GradientBoostingClassifier(random_state=50)
logger.info(model_GB)
logger.info("")
      
# ----------------------------------                                      
# Création du pipeline de traitement des données
# ----------------------------------

log.print_section(logger, "Création pipeline de traitement de données")
                               
# Copie du pipeline commun
# ------------------------
logger.info("...copie du pipeline commun de création des features (cf. 002_preprocessing)")
pipeline_GB = deepcopy(pipeline_common)

# pipeline spécifique pour le modèle
# ----------------------------------
# . toute variable catégorielle : OHE (contrainte sklearn)
# . toute variable numérique : passthrough (pas de contrainte de mise à l'échelle)
#
logger.info("...ajout du preprocessing dédié au modèle GB")

preprocessing_GB = ColumnTransformer(
    transformers=[
        ('numeric', "passthrough", make_column_selector(dtype_include='number')),
        ('category', OneHotEncoder(sparse_output=False, drop=None, handle_unknown='ignore'), make_column_selector(dtype_include=['category','bool'])),
    ], 
    remainder="passthrough", 
    verbose_feature_names_out=False
)

pipeline_GB.steps.append(("preprocessing", preprocessing_GB))

# ajout de l'estimateur
# ---------------------
logger.info("...ajout de l'estimateur GB")
pipeline_GB.steps.append(("estimator", model_GB))

logger.info("")
logger.info(f'------- Pipeline -------')
logger.info(pipeline_GB)
logger.info("")
                                      

# -------------------------
# Création de la grille de méta-paramètres
# -------------------------
  
log.print_section(logger, "Création de la grille")
grid_GB = {     
    'estimator__n_estimators': range(100, 500, 100),
    'estimator__learning_rate': [0.2, 0.3, 0.4, 0.5],
    'estimator__max_depth': [2, 3, 4],
}
logger.debug(grid_GB)
logger.debug("")

# --------------------
# Ajustement du modèle par GSCV
# --------------------
# . Données : pipeline feature + preprocessing DT
# . application des méta-paramètres de la grille définie
# . politique commune de mesure des performance (resampling K-Fold à 5 blocs)
# 

log.print_section(logger, "Création de GSCV")
GSCV_GB = GridSearchCV( estimator=pipeline_GB, 
                        param_grid=grid_GB,
                        cv=resample,
                        scoring="accuracy", 
                        refit=True,
                        n_jobs=-1)
logger.info("")
logger.debug(GSCV_GB)
logger.info("")

logger.info("... ajustement du GSCV")
optimised_GB = GSCV_GB.fit(X_train, y_train)
logger.info("")

# Meilleur modèle
# ---------------
log.print_section(logger, "RESULTATS : meilleur modèle")
logger.debug(f'paramètres: {optimised_GB.best_params_}')
logger.info("")

# Scores par blocs / moyen
# ------------------------
cv_data = ['mean_test_score', 'std_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score','split4_test_score']
logger.debug(f'scores by split:')
for cv in cv_data:
  logger.debug(f'{cv:>20}: {optimised_GB.cv_results_[cv][optimised_GB.best_index_]:02f}')
logger.info("")  

# Scores
# ------
logger.debug(f'score sur X_Train: {accuracy_score(y_train, optimised_GB.predict(X_train))}')
  

# ==============================
# SERIALISATION du MODELE AJUSTE
# ==============================

log.print_section(logger, "Sérialisation du modèle ajusté")

# Sauvegarde du modèle ajusté
# ---------------------------
logger.info("003_main_fitted_gradient_boosting.joblib")
dump(optimised_GB, "./models/003_main_fitted_gradient_boosting.joblib")
logger.info("")  
# %%
