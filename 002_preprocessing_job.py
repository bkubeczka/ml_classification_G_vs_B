# %% LIBRARIES

import pandas as pd
from copy import deepcopy
from joblib import load, dump
import toolbox_constantes as cst
import logging
import toolbox_logging as log
import toolbox_feature_engineering as fe_main
import toolbox_feature_engineering_job as fe_job

#%% LOGGING

logging_module_name = "002_preprocessing_job"
logging_level = logging.INFO

logger = log.getLogger(logging_module_name, logging_level)

log.print_phase(logger, "002. CONSTRUCTION des FEATURES du DATASET JOB (preprocessing)")

# %% 

# =======================================
# Désérialisation du jeu de données MAIN_X_TRAIN et MAIN_y_TRAIN
# =======================================

log.print_section(logger, "Désérialisation des datasets")

# IMPORTANT :
# - La performance des modèles dédiés au jeu de données JOB sera évalué avec les jeux MAIN_X_EVAL / MAIN_Y_EVAL
#   cf. 005_predictions.py
# - pour éviter les fuites de données (i.e. ne pas apprendre sur des observations du jeu d'évaluation MAIN_X_EVAL)
#   les modèles dédiés au jeu de données JOB vont être appris à partir des observations de MAIN_X_TRAIN
#   (plutôt que MAIN_LEARN_X_RAW) issu du préprocessing MAIN (cf. 002_preprocessing_MAIN.py)

logger.info(f"  fichier: {cst.JOBLIB_002_MAIN_X_TRAIN}")
main_X_train = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_X_TRAIN}")

logger.info(f"  fichier: {cst.JOBLIB_002_MAIN_Y_TRAIN}")
main_y_train = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_Y_TRAIN}")

# %% 
#
# ==========================================
# FILTRAGE : FILTRAGE des INDIVIDUS de MAIN : MAINTIEN des INDIVIDUS dans JOB UNIQUEMENT (basé sur la colonne job_category)
# ==========================================

job_X_train = main_X_train.loc[main_X_train.job_category.isna()==False]
job_y_train = main_y_train.loc[job_X_train.index]

assert (~job_X_train.job_category.isna()).all()==True # pas de données manquantes dans la colonne Job_Category 
assert (job_X_train.index==job_y_train.index).all()==True # les index de X et y sont identiques 

logger.debug("jeu de données train:")
logger.debug(f'shape job_X_train: {job_X_train.shape}')
logger.debug(f'shape job_y_train: {job_y_train.shape}')
logger.debug(f'{job_y_train.value_counts()/job_y_train.size}\n')
assert (job_X_train.index==job_y_train.index).all()==True # Jeu de données JOB enrichi : les index de X_train et y_train sont identiques 

logger.info("")

# %%
#
# =======================
# FEATURE ENGINEERING JOB : création du pipeline commun à tous les modèles
# =======================
#

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector


log.print_section(logger, "Pipeline commun")

# COMMON PIPELINE JOB
# ===================
#
# Transformations applicables à tous les modèles
# 

columns_numeric = ['AGE_2019', 'TOWN_TYPE_', 'DISTANCE_GRANDE_VILLE_', # colonnes du jeu de données principal
                   'Working_hours', 'EMOLUMENT' # colonnes du jeu de données JOB
                  ]
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
    'employer_category',
    'FONCTION_PUBLIQUE_',
    'job_category',
    'EMPLOYEE_COUNT', 
    'Terms_of_emp',
    'Eco_sect',
    'work_description',
    'WD_N1_',
    'WD_N2_',
    'Job_dep',
    'JOB_REGION_',
    'WORK_CONDITION',
]

logger.info("")
logger.info(f'colonnes numériques : {columns_numeric}')
logger.info(f'colonnes catégorielles : {columns_category}')
logger.info(f'Les colonnes non listées sont supprimées')
logger.info("")

# Création de nouvelles features
# depuis le Dataset principal (xxx_dataset)
# ------------------------
# cf. rapport pour les transformations appliquées

features_common = ColumnTransformer(
    transformers=[
        ("insee_code", FunctionTransformer(fe_main.insee_code_fe, validate=False), 'insee_code'),
        ("OCCUPATION_42", FunctionTransformer(fe_main.OCCUPATION_42_fe, validate=False), 'OCCUPATION_42'),
        ("ACTIVITY_TYPE", FunctionTransformer(fe_main.ACTIVITY_TYPE_fe, validate=False), 'ACTIVITY_TYPE'),
        ("household", FunctionTransformer(fe_main.household_fe, validate=False), 'household'),
        ("Highest_degree", FunctionTransformer(fe_main.Highest_degree_fe, validate=False), 'Highest_degree'),
        ("sex", FunctionTransformer(fe_main.sex_fe, validate=False), 'sex'),
        ("Others", "passthrough", ['is_student','AGE_2019']),
        # dataset Sport
        ("Club", FunctionTransformer(fe_main.Club_fe, validate=False), 'Club'),
        # dataset Emp_contract
        ("Emp_contract", FunctionTransformer(fe_main.Emp_contract_fe, validate=False), ['Emp_contract', 'ACTIVITY_TYPE']),
        # dataset JOB
        ('employer_category', FunctionTransformer(fe_job.employer_category_fe, validate=False), ['employer_category']),
        ('job_category', FunctionTransformer(fe_job.job_category_fe, validate=False), ['job_category']),
        ('Terms_of_emp', FunctionTransformer(fe_job.terms_of_emp_fe, validate=False), ['Terms_of_emp']),
        ('Eco_sect', FunctionTransformer(fe_job.eco_sect_fe, validate=False), ['Eco_sect']),
        ('work_description', FunctionTransformer(fe_job.work_description_fe, validate=False), ['work_description']),
        ('Job_dep', FunctionTransformer(fe_job.job_dep_fe, validate=False), ['Job_dep']),
        ('WORK_CONDITION', FunctionTransformer(fe_job.work_condition_fe, validate=False), ['WORK_CONDITION']),
        ('EMPLOYEE_COUNT', FunctionTransformer(fe_job.employee_count_fe, validate=False), ['EMPLOYEE_COUNT']), 
        ('Working_hours', FunctionTransformer(fe_job.working_hours_fe, validate=False), ['Working_hours']), 
        ('EMOLUMENT', FunctionTransformer(fe_job.emolument_fe, validate=False), ['EMOLUMENT']),        
        # suppression Id
        ("Id", "drop", 'Id'),        
    ], 
    remainder="drop", 
    verbose_feature_names_out=False
)

features_common.set_output(transform="pandas")

job_pipeline_common = Pipeline (
    steps=[ 
            ('features', features_common), 
    ]
)

# TEST du PIPELINE sur job_X_TRAIN
# ================================

# Copie du pipeline
check_pipeline = deepcopy(job_pipeline_common)

# Application du pipeline sur X_train
# -----------------------------------
logger.info("... Application du pipeline à job_X_train pour vérification (pipeline_common.fit)")
check_X = check_pipeline.fit_transform(
    X=job_X_train,
    y=job_y_train
)

# Vérification de la shape (nombre observation)
logger.info(f'shape : {check_X.shape}')
logger.info("")

# Vérification du maintien des index
logger.info("... Vérification du maintien des index du Dataframe")
logger.debug(f'index X_train initiaux : {job_X_train.index}')
logger.debug(f'index y_train initiaux : {job_y_train.index}')
logger.debug(f'index X transformés : {check_X.index}')
logger.info(f'Conservation des index des X avant et après preprocessing : {(check_X.index==job_X_train.index).all().all()}')
logger.info(f'Conservation des index entre y et X après preprocessing : {(check_X.index==job_y_train.index).all().all()}')
logger.info("")

# validation des 1ère valeurs
logger.info("... Validation des features générées")
logger.info(f'{check_X.head(5)}')
logger.info("")

# validation des types
logger.info("... Validation des colonnes et leur typage")
logger.info(f'{check_X.info()}')
logger.info("")


# Données manquantes
# ------------------
logger.info("... vérification des données manquantes")
logger.info(f'{check_X.isna().all()}')
logger.info('')

# assertions
# ----------
assert (~check_X.job_category.isna()).all()==True # Pas de données manquantes dans job_category (issu du jeu de données JOB)
assert (check_X.index==job_X_train.index).all().all()==True # maintien des index des X avant et après preprocessing
assert (check_X.index==job_y_train.index).all().all()==True # maintien des index entre y et X après preprocessing
assert (~check_X.isna()).all().all()==True # pas de données manquantes à la sortie du preprocessing


# %% 
#
# ========================================
# STRATEGIE d'ESTIMATION de la PERFORMANCE : REECHANTILLONNAGE
# ========================================

from sklearn.model_selection import KFold

log.print_section(logger, "Resample : stratégie d'évaluation des modèles candidats", ln=False)

job_resample = KFold(n_splits=5, shuffle=True, random_state=50)
logger.info(job_resample)
logger.info('')

# %%
# =================================
# Serialisation des données pour la construction des modèles
# =================================

log.print_section(logger, "Sérialisation des objets")

# jeux de données train pour la construction et l'évaluation des modèles JOB
dump(job_X_train, f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_X_TRAIN}")
dump(job_y_train, f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_Y_TRAIN}")

logger.info(f"")
logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_X_TRAIN}")
logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_Y_TRAIN}")
logger.info(f"")

# pipeline commun pour les modèles JOB
dump(job_pipeline_common, f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_COMMON_PIPELINE}")

logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_COMMON_PIPELINE}")

# Stratégie d'évaluation des modèles JOB candidats
dump(job_resample, f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_RESAMPLE}")

logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_RESAMPLE}")
logger.info(f"")


# %%
