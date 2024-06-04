# %% LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from joblib import dump
import toolbox_constantes as cst
import toolbox_make_dataset as makeds
import toolbox_logging as log
import logging

# %% LOGGING

logging_module_name = "001_make_dataset"
logging_level = logging.INFO

logger = log.getLogger(logging_module_name, logging_level)


log.print_phase(
    logger, f"{logging_module_name} : CREATION des jeux d'apprentissage learn_X_raw et learn_y_raw"
)   
    
# *********************************************
# 1 - DATASET principal + EMP_CONTRACT + SPORT + JOB
# *********************************************

# %%
# ====================================================================
# Chargement des fichiers CSV learn_dataset + données complémentaires
# ====================================================================

log.print_section(logger, "Chargement des fichiers <learn_dataset_xxx.csv>")
logger.info(f"")

# fichier de données principal
# ----------------------------
filename = f"{cst.DATA_DIR}/{cst.LEARN_DATASET}"
learn_dataset_raw = pd.read_table(filename, sep=",")

logger.info(f"{'fichier' : >10} : {filename}")
logger.info(f"{'shape' : >10} : {learn_dataset_raw.shape}")
logger.info(f"")

# fichier de données learn_dataset_emp_contract
# ---------------------------------------------
filename = f"{cst.DATA_DIR}/{cst.LEARN_DATASET_EMP_CONTRACT}"
learn_dataset_emp_contract_raw = pd.read_table(filename, sep=",")

logger.info(f"{'fichier' : >10} : {filename}")
logger.info(f"{'shape' : >10} : {learn_dataset_emp_contract_raw.shape}")
logger.info(f"")

# fichier de données learn_dataset_sport
# --------------------------------------
filename = f"{cst.DATA_DIR}/{cst.LEARN_DATASET_SPORT}"
learn_dataset_sport_raw = pd.read_table(filename, sep=",")

logger.info(f"{'fichier' : >10} : {filename}")
logger.info(f"{'shape' : >10} : {learn_dataset_sport_raw.shape}")
logger.info(f"")

# fichier de données train_dataset_job
# ------------------------------------
filename = f"{cst.DATA_DIR}/{cst.LEARN_DATASET_JOB}"
learn_dataset_job_raw = pd.read_table(filename, sep=",")

logger.info(f"{'fichier' : >10} : {filename}")
logger.info(f"{'shape' : >10} : {learn_dataset_job_raw.shape}")
logger.info(f"")


# %%
# =================================
# 1.2 Création d'un jeu de données UNIQUE à partir des fichiers sources learn_dataset_xxx.csv
# =================================
#

log.print_section(logger, "Construction du jeux de données UNIQUE learn_xxx")
logger.info(f"")


# Jeu de données MAIN
# -------------------
df_learn_raw = makeds.make_dataset_main(
    ds_main=learn_dataset_raw,
    ds_job=learn_dataset_job_raw,
    ds_emp_contract=learn_dataset_emp_contract_raw,
    ds_sport=learn_dataset_sport_raw
)

logger.info(f"Jeu de données LEARN après agrégation des données :")
logger.info(f"")

logger.info(f"------- Colonnes et Types ------")
logger.info(f"")
logger.info(f"\n{learn_dataset_raw.info()}")
logger.info(f"")

logger.info(f"------- Données -----")
logger.info(f"")
logger.info(f"\n{learn_dataset_raw.head()}")
logger.info(f"")


# %%
# =================================
# 1.3. Séparation FEATURES / LABELS
# =================================
"""_summary_
Objectifs
- Séparer features et target des jeux de données d'apprentissage MAIN et JOB
"""

logger.info("")
log.print_section(logger, "Séparation Features / Labels")

df_learn_X_raw = df_learn_raw.drop("target", axis=1)
df_learn_y_raw = df_learn_raw.target

logger.info(f'{"learn_X_raw shape" : >15} : {df_learn_X_raw.shape}')
logger.info(f'{"learn_y_raw shape" : >15} : {df_learn_y_raw.shape}')
logger.info(f"")



# %%
# =====================================
# 1.4 Serialisation des jeux de données
# =====================================

"""_summary_
Objectifs
- Stocker les datasets pour utilisation dans les étapes suivantes
"""

log.print_section(logger, "Sérialisation des dataframes source")

dump(df_learn_X_raw, f"{cst.MODELS_DIR}/{cst.JOBLIB_001_LEARN_X_RAW}")
dump(df_learn_y_raw, f"{cst.MODELS_DIR}/{cst.JOBLIB_001_LEARN_Y_RAW}")

logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_001_LEARN_X_RAW}")
logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_001_LEARN_Y_RAW}")
logger.info(f"")

