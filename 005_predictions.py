# %%
# ****************************
# 005_PREDICTIONS
#
# 1. Chargement du meilleur modèle (identifié dans étape 4)
# 2. Mesure de sa performance sur jeu de données X_eval / y_eval
# 3. Prédiction du jeu de données Test
# 4. Sauvegarde des prédictions dans predictions.csv
#
# ****************************

import logging
from joblib import load, dump
import logging
import toolbox_logging as log
import toolbox_constantes as cst
import toolbox_make_dataset as makeds

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# %%

logging_module_name = "005_predictions"
logging_level = logging.INFO

logger = log.getLogger(logging_module_name, logging_level)

log.print_phase(logger, "005. MESURE de PERFORMANCE / PREDICTIONS")


# %%

# =====================================
# 1. CHARGEMENT du MEILLEUR MODELE sur jeu de données MAIN (tout individu)
# =====================================

log.print_section(logger, "Chargement du meilleur modele MAIN")

# chargement du nom de fichier modèle identifié à l'étape 4
# -----------------------------------
main_best_model_name = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_BEST_MODEL}")

# Chargement du modèle associé
# -----------------------------
main_best_model = load(f"{cst.MODELS_DIR}/{main_best_model_name}")

logger.info(f"best model MAIN : {main_best_model_name}")
logger.info("")

# ====================================
# 2. CHARGEMENT du MEILLEUR MODELE sur le jeu de données JOB (individus avec JOB)
# ====================================

log.print_section(logger, "Chargement du meilleur modele JOB")

# chargement du nom de fichier modèle identifié à l'étape 4
# ---------------------------------------------------------
job_best_model_name = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_JOB_BEST_MODEL}")

# Chargement du modèle associé
# -----------------------------
job_best_model = load(f"{cst.MODELS_DIR}/{job_best_model_name}")

logger.info(f"best model : {job_best_model_name}")
logger.info("")


# ================================================
# 3. CHOIX de la MEILLEURE CONFIGURATION de MODELE : MAIN seul ou MAIN+JOB
# ================================================

# 2 cas possibles :
#
# MODELE_MAIN_ONLY 
# si le score Accuracy du modèle MAIN >= score Accuracy du modèle JOB, (i.e. le modèle MAIN prédit plus de bonnes réponses que JOB)
# alors la prédiction est faîte uniquement sur le meilleur modèle appris sur le jeu de données MAIN
#
# MODELE_COMBINED 
# si le score Accuracy du modèle MAIN < score Accuracy du modèle JOB, (i.e. le modèle JOB prédit plus de bonnes réponses que MAIN sur le périmètre du jeu de données JOB)
# alors la prédiction est faîte d'abord sur le meilleur modèle appris sur le jeu de données MAIN
# puis la prédiction des individus apparaissant dans JOB sont mis à jour par la prédiction du meilleur modèle JOB
#

log.print_section(logger, "Choix de la meilleure configuration de modèles : modèle MAIN seul ou modèles MAIN+JOB combinés")

# Score moyen du meilleur modèle MAIN
# -----------------------------------
main_best_mean_score = main_best_model.cv_results_["mean_test_score"][main_best_model.best_index_]
logger.info(f"Score moyen du meilleur modèle sur jeu de données MAIN : {main_best_mean_score}")

# Score moyen du meilleur modèle JOB
# -----------------------------------
job_best_mean_score = job_best_model.cv_results_["mean_test_score"][job_best_model.best_index_]
logger.info(f"Score moyen du meilleur modèle sur jeu de données JOB : {job_best_mean_score}")

# Choix
# -----
if (main_best_mean_score >= job_best_mean_score):
    
    logger.info(f"Score MAIN >= Score JOB : utilisation du modèle MAIN seul")
    model_configuration = "MAIN_ONLY"
        
else:
    
    logger.info(f"Score JOB > Score MAIN : utilisation des modèles combinés MAIN+JOB")
    model_configuration = "MAIN_JOB_COMBINED"

logger.info("")

# %%

# ===============================================================================
# 4. MESURE de la PERFOMANCE de LA COMBINAISON de MODELE, sur jeu de données EVAL
# ===============================================================================

log.print_section(logger, "Mesure de performance de la combinaison de modèles sur sur jeu de données d'évaluation (main_X_eval / main_y_eval)")

# ----------------------------------------
# 4.1. CHARGEMENT des DONNEES d'EVALUATION (cf. preprocessing_main)
# ----------------------------------------

# Chargement de main_X_eval
# -------------------------
main_X_eval = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_X_EVAL}")
logger.info(f"{'fichier' : >10} : {cst.JOBLIB_002_MAIN_X_EVAL}")
logger.debug(f"X_eval: shape {main_X_eval.shape}")

# Chargement de main_y_eval
# -------------------------
main_y_eval = load(f"{cst.MODELS_DIR}/{cst.JOBLIB_002_MAIN_Y_EVAL}")
logger.info(f"{'fichier' : >10} : {cst.JOBLIB_002_MAIN_Y_EVAL}")
logger.debug(f"y_eval: shape {main_y_eval.shape}")

assert (main_X_eval.index==main_y_eval.index).all() # les index des X et y sont identiques


# ----------------------------------------------------------------
# 4.1. PREDICTIONS avec le meilleur modèle sur jeu de données MAIN (i.e. sans les colonnes JOB)
# ----------------------------------------------------------------

# Prédiction de main_X_eval avec le meilleur modèle MAIN seul
# -----------------------------------------------------------
logger.info("")
logger.info("... predictions de main_X_eval par le meilleur modèle MAIN")
main_y_eval_predictions = pd.Series(main_best_model.predict(main_X_eval))
logger.info(f"    predictions: shape {main_y_eval_predictions.shape}")

assert (main_y_eval_predictions.size==main_X_eval.index.shape[0]) # Vérification du nombre de prédictions
assert (~main_y_eval_predictions.isna()).all() # pas de prédictions manquantes
assert main_y_eval_predictions.unique().size==2 # 2 valeurs possibles uniquement
assert "B" in main_y_eval_predictions.unique()
assert "G" in main_y_eval_predictions.unique()
assert (main_y_eval_predictions.value_counts()["B"] + main_y_eval_predictions.value_counts()["G"]) == main_y_eval_predictions.size

# Application des index d'origine sur les prédictions
# ---------------------------------------------------
main_y_eval_predictions.index = main_X_eval.index

# Mesure de la performance du meilleur modèle MAIN
# ------------------------------------------------
main_score_best_model = accuracy_score(y_true=main_y_eval, y_pred=main_y_eval_predictions)
logger.info(f"    Performance du meilleur modèle MAIN seul ({main_best_model_name}): {main_score_best_model*100: 0.2f}% de bonnes predictions")

# Sérialisation des prédictions MAIN-only
dump(main_y_eval_predictions, f"{cst.MODELS_DIR}/{cst.JOB_005_MAIN_ONLY_EVAL_PREDICTIONS}")
logger.info(f"    Sérialisation : {cst.MODELS_DIR}/{cst.JOB_005_MAIN_ONLY_EVAL_PREDICTIONS}")
logger.info(f"")

# ---------------------------------------------------------------
# 4.2. PREDICTIONS avec le meilleur modèle sur jeu de données JOB (MAIN avec colonnes JOB)
# ---------------------------------------------------------------

if (model_configuration == "MAIN_JOB_COMBINED"):
    
    # FILTRAGE des INDIVIDUS : MAINTIEN des INDIVIDUS avec JOB (basé sur la colonne job_category)
    # ----------------------
    logger.info("... filtrage des individus de main_X_eval (individus avec JOB uniquement)")
    logger.info("")
    job_X_eval = main_X_eval.loc[main_X_eval.job_category.isna()==False]
    job_y_eval = main_y_eval.loc[job_X_eval.index]

    assert (~job_X_eval.job_category.isna()).all()==True # pas de données manquantes dans la colonne Job_Category 
    assert (job_X_eval.index==job_y_eval.index).all()==True # les index de X et y sont identiques 
    
    # Prédiction des individus filtrés avec le meilleur modèle JOB
    # -------------------------------------------------
    logger.info("... predictions de main_X_eval (individus avec JOB uniquement) par le meilleur modèle JOB")
    job_y_eval_predictions = pd.Series(job_best_model.predict(job_X_eval))
    logger.info(f"    predictions: shape {job_y_eval_predictions.shape}")

    assert (job_y_eval_predictions.size==job_X_eval.shape[0]) # Vérification du nombre de prédictions
    assert (~job_y_eval_predictions.isna()).all() # pas de prédictions manquantes
    assert job_y_eval_predictions.unique().size==2 # 2 valeurs possibles uniquement
    assert "B" in job_y_eval_predictions.unique()
    assert "G" in job_y_eval_predictions.unique()
    assert (job_y_eval_predictions.value_counts()["B"] + job_y_eval_predictions.value_counts()["G"]) == job_y_eval_predictions.size
    
    # Application des index d'origine sur les prédictions
    # ---------------------------------------------------
    job_y_eval_predictions.index = job_X_eval.index # Vérification de la correspondance des index entre prédictions et "y à prédire"
    
    # Mesure de la performance du meilleur modèle JOB
    # -----------------------------------------------
    job_score_best_model = accuracy_score(y_true=job_y_eval, y_pred=job_y_eval_predictions)
    logger.info(f"    Performance du meilleur modèle JOB seul ({job_best_model_name}): {job_score_best_model*100: 0.2f}% de bonnes predictions")
    
    # Sérialisation des prédictions JOB-only
    dump(job_y_eval_predictions, f"{cst.MODELS_DIR}/{cst.JOB_005_JOB_ONLY_EVAL_PREDICTIONS}")
    logger.info(f"    Sérialisation : {cst.MODELS_DIR}/{cst.JOB_005_JOB_ONLY_EVAL_PREDICTIONS}")
    logger.info(f"")

    # substituer les prédictions des individus filtrés
    # ------------------------------------------------
    main_y_eval_predictions.loc[job_y_eval_predictions.index] = job_y_eval_predictions
    

# Score (Taux de bonnes prédictions)
# ----------------------------------
score_best_model = accuracy_score(y_true=main_y_eval, y_pred=main_y_eval_predictions)

logger.info("")
logger.info(
    f"PERFORMANCE du MEILLEUR MODELE SELECTIONNE ({model_configuration}): {score_best_model*100: 0.2f}% de bonnes predictions"
)

# Sérialisation des prédictions finales sur main_X_eval
dump(main_y_eval_predictions, f"{cst.MODELS_DIR}/{cst.JOB_005_FINAL_EVAL_PREDICTIONS}")
logger.info(f"    Sérialisation : {cst.MODELS_DIR}/{cst.JOB_005_FINAL_EVAL_PREDICTIONS}")
logger.info(f"")


# Confusion Matrix
# ----------------
logger.info(f"Confusion Matrix")
confusion_matrix = confusion_matrix(y_true=main_y_eval, y_pred=main_y_eval_predictions)
logger.info(f"\n {confusion_matrix}")

# Classification report
# ---------------------
logger.info(f"Classification Report")
report = classification_report(y_true=main_y_eval, y_pred=main_y_eval_predictions)
logger.info(f"\n {report}")


# %%

# ==============================================
# 5. CHARGEMENT du JEU de DONNEES TEST à prédire
# ==============================================

log.print_section(logger, "Chargement des fichiers <test_dataset_xxx.csv>")

# Chargement du jeu de données test_dataset
# -----------------------------------------
test_dataset_raw = pd.read_table(f"{cst.DATA_DIR}/{cst.TEST_DATASET}", sep=",")

logger.info(f"test_dataset: shape {test_dataset_raw.shape}")
logger.debug(
    f"{len(test_dataset_raw.columns)} colonnes:\n\n {test_dataset_raw.columns}\n"
)

# fichier de données xxx_dataset_emp_contract
filename = f"{cst.DATA_DIR}/{cst.TEST_DATASET_EMP_CONTRACT}"
test_dataset_emp_contract_raw = pd.read_table(filename, sep=",")

logger.info(f"")
logger.info(f"{'fichier' : >10} : {filename}")
logger.info(f"{'shape' : >10} : {test_dataset_emp_contract_raw.shape}")
logger.info(f"")

# fichier de données xxx_dataset_job
filename = f"{cst.DATA_DIR}/{cst.TEST_DATASET_JOB}"
test_dataset_job_raw = pd.read_table(filename, sep=",")

logger.info(f"{'fichier' : >10} : {filename}")
logger.info(f"{'shape' : >10} : {test_dataset_job_raw.shape}")
logger.info(f"")

# fichier de données xxx_dataset_sport
filename = f"{cst.DATA_DIR}/{cst.TEST_DATASET_SPORT}"
test_dataset_sport_raw = pd.read_table(filename, sep=",")

logger.info(f"{'fichier' : >10} : {filename}")
logger.info(f"{'shape' : >10} : {test_dataset_sport_raw.shape}")
logger.info(f"")


# ===============================
# 6. Création du jeu de données UNIQUE à partir des données brutes  
# ===============================

log.print_section(logger, "Construction du jeux de données UNIQUE X_test")
logger.info(f"")

main_X_test = makeds.make_dataset_main(
    ds_main=test_dataset_raw,
    ds_job=test_dataset_job_raw,
    ds_emp_contract=test_dataset_emp_contract_raw,
    ds_sport=test_dataset_sport_raw
) 

logger.info(f"X_test : shape {main_X_test.shape}")
logger.info("")

# logger.info(f"{'types des colonnes:' : >20}")
logger.info(f"------- Colonnes et Types ------")
logger.info(f"\n{main_X_test.info()}")
logger.info(f"")

# logger.info(f'données:\n{test_dataset_raw.head()}')
logger.debug(f"------- Données -----")
logger.debug(f"\n{main_X_test.head()}")
logger.debug(f"")


# %%

# =========================================================
# PREDICTIONS du jeu TEST avec le meilleur modèle identifié
# =========================================================

log.print_section(logger, f"Predictions du jeu de données TEST")

# Prédiction de main_X_test avec le meilleur modèle MAIN
# ------------------------------------------------------
logger.info("")
logger.info(f"... Predictions du jeu de données TEST par le meilleur modèle MAIN ({main_best_model_name})")

main_y_test_predictions = pd.Series(main_best_model.predict(main_X_test))
logger.info(f"    X_test      : shape {main_X_test.shape}")
logger.info(f"    predictions : shape {main_y_test_predictions.shape}")
logger.info("")

assert (main_y_test_predictions.shape[0]==main_X_test.shape[0]) # Maintien du nombre d'individus
assert (~main_y_test_predictions.isna()).all() # pas de prédictions manquantes
assert main_y_test_predictions.unique().size==2 # 2 valeurs possibles uniquement
assert "B" in main_y_test_predictions.unique()
assert "G" in main_y_test_predictions.unique()
assert (main_y_test_predictions.value_counts()["B"] + main_y_test_predictions.value_counts()["G"]) == main_y_test_predictions.size

# Application des index d'origine sur les prédictions
# ---------------------------------------------------
main_y_test_predictions.index = main_X_test.index

# Prédiction des individus ayant un JOB avec le meilleur modèle JOB
# -----------------------------------------------------------------
if (model_configuration == "MAIN_JOB_COMBINED"):
    
    # FILTRAGE des INDIVIDUS : MAINTIEN des INDIVIDUS avec JOB (basé sur la colonne job_category)
    # ----------------------
    
    logger.info("... filtrage des individus de X_test (individus avec JOB uniquement)")
    logger.info("")

    job_X_test = main_X_test.loc[main_X_test.job_category.isna()==False]

    assert (~job_X_test.job_category.isna()).all()==True # pas de données manquantes dans la colonne Job_Category 
    
    # Prédiction des individus filtrés avec le meilleur modèle JOB
    # -------------------------------------------------
    logger.info(f"... predictions de X_test (individus avec JOB uniquement) par le meilleur modèle JOB ({job_best_model_name})")
    job_y_test_predictions = pd.Series(job_best_model.predict(job_X_test))
    logger.info(f"    X_test filtré pour JOB : shape {job_X_test.shape}")
    logger.info(f"   predictions JOB : shape {job_y_test_predictions.shape}")
    logger.info("")

    assert (job_y_test_predictions.shape[0]==job_X_test.shape[0]) # Maintien du nombre d'individus
    assert (~job_y_test_predictions.isna()).all() # pas de prédictions manquantes
    assert job_y_test_predictions.unique().size==2 # 2 valeurs possibles uniquement
    assert "B" in job_y_test_predictions.unique()
    assert "G" in job_y_test_predictions.unique()
    assert (job_y_test_predictions.value_counts()["B"] + job_y_test_predictions.value_counts()["G"]) == job_y_test_predictions.size
    
    # Application des index d'origine sur les prédictions
    # ---------------------------------------------------
    job_y_test_predictions.index = job_X_test.index # Vérification de la correspondance des index entre prédictions et "y à prédire"
    
    # substituer les prédictions des individus filtrés
    # ------------------------------------------------
    main_y_test_predictions.loc[job_y_test_predictions.index] = job_y_test_predictions


# Création du Data Frame contenant les prédictions
# ------------------------------------------------
# 2 colonnes
# . Id
# . target
#
df_predictions = pd.DataFrame()
df_predictions["Id"] = main_X_test.Id.values
df_predictions["target"] = main_y_test_predictions.values.astype("str")

# Vérification
# ------------
logger.info(f"predictions: shape {df_predictions.shape}")
logger.info(f"predictions: shape {df_predictions.info()}")
logger.info(
    f"proportion B/G : {df_predictions.target.value_counts() / df_predictions.target.count()}"
)

assert df_predictions.shape[0] == main_X_test.shape[0]  # vérification du nombre d'observations
assert df_predictions.shape[1] == 2  # Vérification du nombre de Colonnes
assert "Id" in df_predictions.columns  # Vérification du nom des Colonnes
assert "target" in df_predictions.columns  # Vérification du nom des Colonnes
assert (df_predictions.Id!=main_X_test.Id).sum() == 0  # ordre des observations identique
assert df_predictions.target.isna().sum() == 0  # pas de valeurs manquantes
assert df_predictions.target.unique().size == 2  # 2 modalités uniquement
assert "B" in df_predictions.target.unique()  # B est une modalité
assert "G" in df_predictions.target.unique()  # G est une modalité


# %%

# ==========================
# SAUVEGARDE predictions.csv
# ==========================
import csv

log.print_section(logger, f"Sauvegarde des prédictions")

df_predictions.to_csv(
    "./predictions.csv",
    header=True,
    quotechar='"',
    quoting=csv.QUOTE_NONNUMERIC,
    index=False,
)

logger.info("./predictions.csv")

# %%

# Serialisation des résultats
dump(df_predictions, f"{cst.MODELS_DIR}/{cst.JOBLIB_005_FINAL_TEST_PREDICTIONS}")
logger.info(f"{cst.MODELS_DIR}/{cst.JOBLIB_005_FINAL_TEST_PREDICTIONS}")
logger.info(f"")

