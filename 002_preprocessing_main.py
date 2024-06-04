# %% LIBRARIES

import pandas as pd
from copy import deepcopy
from joblib import load, dump
import toolbox_constantes as cst
import toolbox_logging as log
import toolbox_feature_engineering as fe

#%% LOGGING

import logging

logging_module_name = "002_preprocessing_main"
logging_level = logging.INFO

logger = log.getLogger(logging_module_name, logging_level)

log.print_phase(logger, "002. CONSTRUCTION des FEATURES pour le jeu de données MAIN (preprocessing)")

# %% 

# ==========================================
# 1. Désérialisation du jeu de données LEARN
# ==========================================

log.print_section(logger, "Désérialisation des dataframes MAIN")
logger.info(f"")

logger.info(f"  {'fichier' : >10} : {cst.JOBLIB_001_LEARN_X_RAW}")
learn_X_raw = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_001_LEARN_X_RAW}")
logger.info(f"{'shape' : >10} : {learn_X_raw.shape}")
logger.info(f"")


logger.info(f"  {'fichier' : >10} : {cst.JOBLIB_001_LEARN_Y_RAW}")
learn_y_raw = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_001_LEARN_Y_RAW}")
logger.info(f"{'shape' : >10} : {learn_y_raw.shape}")
logger.info(f"")


# %% 
#
# =========================================
# 2. SEGMENTATION du JEU de DONNEES "LEARN" en TRAIN 80 / EVAL 20
# =========================================

log.print_section(logger, "Séparation du dataset learn (train/eval)")
logger.info(f"")

logger.info(f"ratio train_size : {cst.LEARN_TRAIN_SIZE}")
logger.info(f"")

# SEPARATION TRAIN / EVAL depuis LEARN
# ------------------------------------
#
# main_X_train, main_y_train : jeux de données dédiés à l'apprentissage des modèles
# main_X_eval, main_y_eval : jeux de données dédiés à l'évaluation de la performance 
#                   des modèles candidats (MAIN et JOB) cf. fichier 005_predictions.py
#
# Ratio train/eval = 80/20
#

from sklearn.model_selection import train_test_split

main_X_train, main_X_eval, main_y_train, main_y_eval = train_test_split(
                                                    learn_X_raw, learn_y_raw, 
                                                    train_size=cst.LEARN_TRAIN_SIZE, # cf. définition dans toolbox_constantes.py
                                                    shuffle=True, 
                                                    random_state=50, 
                                                    stratify=learn_y_raw)

logger.info("**jeux de données train**")
logger.info(f"")
logger.info(f'main_X_train: {main_X_train.shape}')
logger.info(f'main_y_train: {main_y_train.shape}')
logger.info(f'{main_y_train.value_counts()/main_y_train.size}')
logger.info(f"")

logger.info("**jeux de données eval**")
logger.info(f"")
logger.info(f'main_X_eval: {main_X_eval.shape}')
logger.info(f'main_y_eval: {main_y_eval.shape}')
logger.info(f'{main_y_eval.value_counts()/main_y_eval.size}\n')
logger.info(f"")

# %%
#
# ===================
# 3. FEATURE ENGINEERING : création du pipeline commun à tous les modèles
# ===================
#

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector

log.print_section(logger, "Pipeline commun pour le jeu de données MAIN")
logger.info(f"")

# COMMON PIPELINE
# ===============
#
# Transformations applicables à tous les modèles
# 

columns_numeric = ['AGE_2019', 'TOWN_TYPE_', 'DISTANCE_GRANDE_VILLE_']
columns_category = [
    # depuis insee_code
    'is_student', "sex",
    'DEPARTEMENT_', 'REGION_',
    # depuis Id
    'CSP_N1_', 'CSP_N2_', 
    'ACTIVITY_L1_', 'ACTIVITY_L2_',
    'HOUSEHOLD_L1_', 'HOUSEHOLD_L2_',
    'DEGREE_',
    # sport
    'SPORTIF_',
    'HANDICAP_',
    # Emp_contract
    'CONTRAT_',
    # Job 
    # 'employer_category',
    # 'job_category',
    # 'EMPLOYEE_COUNT',
    # 'Terms_of-emp',
    # 'Eco_sect',
    # 'work_description',
    # 'Job_dep',
    # 'Working_hours',
    # 'WORK_CONDITION',
    # 'EMOLUMENT'
]

logger.info(f'colonnes numériques : {columns_numeric}')
logger.info(f'colonnes catégorielles : {columns_category}')
logger.info(f'Les colonnes non listées sont supprimées (Id + colonnes issues de JOB)')
logger.info("")

# Création de nouvelles features pour le jeu MAIN
# ------------------------
# cf. rapport pour les transformations appliquées

features_common = ColumnTransformer(
    transformers=[
        ("insee_code", FunctionTransformer(fe.insee_code_fe, validate=False), 'insee_code'),
        ("OCCUPATION_42", FunctionTransformer(fe.OCCUPATION_42_fe, validate=False), 'OCCUPATION_42'),
        ("ACTIVITY_TYPE", FunctionTransformer(fe.ACTIVITY_TYPE_fe, validate=False), 'ACTIVITY_TYPE'),
        ("household", FunctionTransformer(fe.household_fe, validate=False), 'household'),
        ("Highest_degree", FunctionTransformer(fe.Highest_degree_fe, validate=False), 'Highest_degree'),
        ("sex", FunctionTransformer(fe.sex_fe, validate=False), 'sex'),
        ("Others", "passthrough", ['is_student','AGE_2019']),
        # dataset Sport
        ("Club", FunctionTransformer(fe.Club_fe, validate=False), 'Club'),
        # dataset Emp_contract
        ("Emp_contract", FunctionTransformer(fe.Emp_contract_fe, validate=False), ['Emp_contract', 'ACTIVITY_TYPE']),
        # suppression Id
        ("Id", "drop", 'Id'), 
    ], 
    remainder="drop", # suppression des colonnes issues JOB
    verbose_feature_names_out=False
)

features_common.set_output(transform="pandas")

main_pipeline_common = Pipeline (
    steps=[ 
            ('features', features_common), 
    ]
)

# VALIDATION du FONCTIONNEMENT du PIPELINE
# ========================================

log.print_section(logger, "Validation du Pipeline commun avec le jeu de données MAIN TRAIN")
logger.info(f"")

# Copie du pipeline sur X_train
# -----------------------------
logger.info("... Création d'une copie du pipeline commun (deepcopy)")
check_pipeline_common = deepcopy(main_pipeline_common)

# Application du pipeline sur X_train
# -----------------------------------
logger.info("... Application du pipeline à X_train (pipeline_common.fit)")
check_X = check_pipeline_common.fit_transform(
    X=main_X_train,
    y=main_y_train
)

# Vérification de la shape (nombre observation)
logger.info("... Vérification de la sortie")
logger.info(f'    shape : {check_X.shape}')

# Vérification du maintien des index
logger.info("... Vérification du maintien des index du Dataframe")
logger.debug(f'   index X_train initiaux : {main_X_train.index}')
logger.debug(f'   index y_train initiaux : {main_y_train.index}')
logger.debug(f'   index X transformés : {check_X.index}')
logger.info(f'    Maintien des index X_train : {(check_X.index==main_X_train.index).size==main_X_train.index.shape[0]}')
logger.info(f'    Maintien des index y_train : {(check_X.index==main_y_train.index).size==main_y_train.index.shape[0]}')
logger.info("")

# validation des 1ère valeurs
logger.info("... Validation des features générées")
logger.info(f'{check_X.head(5)}')
logger.info("")

# validation des types
logger.info("... Validation du typage")
logger.info(f'{check_X.info()}')
logger.info("")

# Données manquantes
# ------------------
logger.info("... vérification des données manquantes par colonnes")
logger.info(f'{check_X.isna().sum()}')
logger.info('')


# assertions
assert (check_X.index==main_X_train.index).all()==True # maintien des index des X avant et après preprocessing
assert (check_X.index==main_y_train.index).all()==True # maintien des index des y et des X après preprocessing
assert (check_X.isna().sum()).sum()==0 # pas de données manquantes à la sortie du preprocessing




# %% 
#
# ========================================
# STRATEGIE d'ESTIMATION de la PERFORMANCE : REECHANTILLONNAGE
# ========================================

from sklearn.model_selection import KFold

log.print_section(logger, "Resample : stratégie d'évaluation des modèles candidats pour le jeu de données MAIN", ln=False)
logger.info(f"")

main_resample = KFold(n_splits=5, shuffle=True, random_state=50)
logger.info(main_resample)
logger.info('\n')

# %%
# =================================
# Serialisation des données pour la construction des modèles
# =================================

log.print_section(logger, "Sérialisation des objets")
logger.info(f"")

# jeux de données train / test pour la construction et l'évaluation des modèles
dump(main_X_train, f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_X_TRAIN}")
dump(main_y_train, f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_Y_TRAIN}")
dump(main_X_eval, f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_X_EVAL}")
dump(main_y_eval, f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_Y_EVAL}")

logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_X_TRAIN}")
logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_Y_TRAIN}")
logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_X_EVAL}")
logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_Y_EVAL}")
logger.info(f"")

# pipeline commun
dump(main_pipeline_common, f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_COMMON_PIPELINE}")

logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_COMMON_PIPELINE}")
logger.info(f"")

# Stratégie d'évaluation des modèles candidats
dump(main_resample, f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_RESAMPLE}")

logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_RESAMPLE}")
logger.info(f"")

# %%
