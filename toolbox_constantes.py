
# ********************
# CONSTANTES du PROJET
# ********************

# Définition de la taille du jeu de données Train dans l'appel à la fonction TrainTestSplit
# - 0.8 pour travailler sur tout le jeu de donnée 
# - 0.2 pour la phase d mise au point 
# -------------------------- 
# LEARN_TRAIN_SIZE = 0.2 # Phase de mise au point
LEARN_TRAIN_SIZE = 0.8 # Entraînement final


# répertoires
# ===========
DATA_DIR = "./data"
LOGS_DIR = "./logs"
MODELS_DIR = "./models"
REPORTS_DIR = "./reports"

# Fichiers learn au format CSV
# ============================
LEARN_DATASET = "learn_dataset.csv"
LEARN_DATASET_EMP_CONTRACT = "learn_dataset_Emp_contract.csv"
LEARN_DATASET_JOB = "learn_dataset_job.csv"
LEARN_DATASET_SPORT = "learn_dataset_sport.csv"

# Fichiers test au format CSV
# ===========================
TEST_DATASET = "test_dataset.csv"
TEST_DATASET_EMP_CONTRACT = "test_dataset_Emp_contract.csv"
TEST_DATASET_JOB = "test_dataset_job.csv"
TEST_DATASET_SPORT = "test_dataset_sport.csv"

# Jeu de données source UNIQUE
# ============================
JOBLIB_001_LEARN_X_RAW = "001_learn_X_raw.joblib"
JOBLIB_001_LEARN_Y_RAW = "001_learn_y_raw.joblib"

# fichiers intermédiaires sérialisés : apprentissage MAIN
# ========================================================
JOBLIB_002_MAIN_X_TRAIN = "002_main_X_train.joblib"
JOBLIB_002_MAIN_Y_TRAIN = "002_main_y_train.joblib"
JOBLIB_002_MAIN_X_EVAL = "002_main_X_eval.joblib"
JOBLIB_002_MAIN_Y_EVAL = "002_main_y_eval.joblib"
JOBLIB_002_MAIN_COMMON_PIPELINE = "002_main_common_pipeline.joblib"
JOBLIB_002_MAIN_RESAMPLE = "002_main_resample.joblib"

JOBLIB_002_MAIN_BEST_MODEL = "004_main_best_model.joblib"

# fichiers intermédiaires sérialisés : apprentissage JOB
# =======================================================

JOBLIB_002_JOB_X_TRAIN = "002_job_X_train.joblib"
JOBLIB_002_JOB_Y_TRAIN = "002_job_y_train.joblib"
JOBLIB_002_JOB_COMMON_PIPELINE = "002_job_common_pipeline.joblib"
JOBLIB_002_JOB_RESAMPLE = "002_job_resample.joblib"

JOBLIB_002_JOB_BEST_MODEL = "004_job_best_model.joblib"

# fichiers intermédiaires sérialisés : Mesure de performance
# ==========================================================

JOB_005_MAIN_ONLY_EVAL_PREDICTIONS = "005_main_only_eval_predictions.joblib"
JOB_005_JOB_ONLY_EVAL_PREDICTIONS = "005_job_only_eval_predictions.joblib"
JOB_005_FINAL_EVAL_PREDICTIONS = "005_final_eval_predictions.joblib"

# fichier de prédiction
# =====================

JOBLIB_005_FINAL_TEST_PREDICTIONS = "005_final_test_predictions.joblib"