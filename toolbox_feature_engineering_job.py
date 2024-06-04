import pandas as pd
import numpy as np
import datetime
import os
import toolbox_constantes as cst



# *******************
# FEATURE ENGINEERING
# *******************

# ====================
# employer_category_fe
# ====================
def employer_category_fe(employer_category: pd.Series) -> pd.DataFrame:
    """employer_category_fe : traitement de la feature employer_category

    - employer_category_fe est de la forme ct_a où a est un entier de 1 à 9
    - les données manquantes (rares) sont imputées par une catégorie fictive "ct_0"
    - création de FONCTION_PUBLIQUE_ (booléenne) : Si la catégorie vaut ct_1 à ct_5
    - maintien de employer_category (category) : catégorie d'employeur

    Args:
        employer_category (Series): Colonne employer_category.

    Returns:
        Dataframe contenant FONCTION_PUBLIQUE_ / employer_category_
    """
    
    # copie de la feature d'origine
    # -----------------------------    
    df = employer_category
    
    # Imputation des données manquantes
    # ---------------------------------
    df.loc[df.employer_category.isna(), "employer_category"] = "ct_0"
    
    # Extraction du code de catégorie
    # -------------------------------

    # pattern regex pour ct_a
    pattern = r"ct_(\d+)"

    # Extraction de x et y
    extracted_values = df["employer_category"].str.extract(pattern, expand=True)
    extracted_values.columns = ["a"]  # Renommer les colonnes
    extracted_values.a = extracted_values.a.astype("int64")

    # Création de la feature FONCTION_PUBLIQUE_
    # ------------------------------
    df["FONCTION_PUBLIQUE_"] = (extracted_values.a <= 5) & (extracted_values.a > 0) # Les catégories inférieures ou égales à 5 sont liées à la fonction publique
    
    # Typage
    # ------
    df["FONCTION_PUBLIQUE_"] = df["FONCTION_PUBLIQUE_"].astype("boolean")
    df["employer_category"] = df["employer_category"].astype("category")
    
    # Validation des sorties
    # ----------------------
    assert df.shape[0] == employer_category.shape[0]  # nombre d'observations
    assert df.shape[1] == 2  # 2 features en sortie
    assert (df.index == employer_category.index).all() == True  # conservation des index avant et après le preprocessing
    assert "FONCTION_PUBLIQUE_" in df.columns.values  # présence FONCTION_PUBLIQUE_
    assert "employer_category" in df.columns.values  # présence employer_category
    assert (~df.isna()).all().all()==True  # pas de données manquantes

    
    return df
    
    
# ===============
# job_category_fe
# ===============
def job_category_fe(job_category: pd.Series) -> pd.DataFrame:
    """job_category_fe : traitement de la feature job_category

    - employer_category_fe est de la forme ct_a où a est un entier de 1 à 9
    - maintien de job_category (category) : "O", "A" ou "X

    Args:
        job_category (Series): Colonne job_category.

    Returns:
        Dataframe contenant job_category
    """
    
    # copie de la feature job_category
    # --------------------------------
    df = job_category
    
    # typage
    # ------
    df.job_category = df.job_category.astype("category")
    
    # Validation des sorties
    # ----------------------
    assert df.shape[0] == job_category.shape[0]  # nombre d'observations
    assert df.shape[1] == 1  # 1 feature en sortie
    assert (df.index == job_category.index).all() == True  # conservation des index avant et après le preprocessing
    assert "job_category" in df.columns.values  # présence DEGREE_
    assert (~df.isna()).all().all()==True  # pas de données manquantes
    
    return df

# =================
# employee_count_fe
# =================
def employee_count_fe(employee_count: pd.Series) -> pd.DataFrame:
    """employee_count_fe : traitement de la feature EMPLOYEE_COUNT

    - employee_count est de la forme tr_x où x est un entier de 0 à 6 selon la taille de l'entreprise
    - maintien de employee_count (category) : code selon la taille de l'entreprise
    - les données manquantes (rares) sont imputées par la catégorie majoritaire  (BKU - revoir)

    Args:
        employee_count (Series): Colonne EMPLOYEE_COUNT.

    Returns:
        Dataframe contenant employee_count
    """
    
    # copie de la feature
    # -------------------
    df = employee_count
    
    # Imputation des données manquantes (BKU : à retravailler par KNN)
    # ---------------------------------
    categorie_majoritaire = df.EMPLOYEE_COUNT.value_counts().index[0]
    df.loc[df.EMPLOYEE_COUNT.isna(), "EMPLOYEE_COUNT"] = categorie_majoritaire
    
    # Typage
    df.EMPLOYEE_COUNT = df.EMPLOYEE_COUNT.astype("category")
    
    # Validation des sorties
    # ----------------------
    assert df.shape[0] == employee_count.shape[0]  # nombre d'observations
    assert df.shape[1] == 1  # 1 feature en sortie
    assert (df.index == employee_count.index).all() == True  # conservation des index avant et après le preprocessing
    assert "EMPLOYEE_COUNT" in df.columns.values  # présence EMPLOYEE_COUNT
    assert (~df.isna()).all().all()==True  # pas de données manquantes
    
    return df

# ===============
# terms_of_emp_fe
# ===============
def terms_of_emp_fe(terms_of_emp: pd.Series) -> pd.DataFrame:
    """terms_of_emp_fe : traitement de la feature terms_of_emp

    - terms_of_emp est fourni sous la forme d'un code en 3 lettres 
    - maintien de terms_of_emp (category) 
    - les données manquantes (rares) sont imputées par la catégorie majoritaire de Terms_of_emp parmi les même triplets employer_category, job_category, EMPLOYEE_COUNT (BKU - revoir)

    Args:
        terms_of_emp (Series): Colonne Terms_of_emp.

    Returns:
        Dataframe contenant Terms_of_emp
    """
    
    # Copie de la feature Terms_of_emp
    # --------------------------------
    df = terms_of_emp
    
    # Imputation des données manquantes (BKU : à retravailler)
    # ---------------------------------
    categorie_majoritaire = df.Terms_of_emp.value_counts().index[0]
    df.loc[df.Terms_of_emp.isna(), "Terms_of_emp"] = categorie_majoritaire
    
    # Typage
    # ------
    df.Terms_of_emp = df.Terms_of_emp.astype("category")
    
    # Validation des sorties
    # ----------------------
    assert df.shape[0] == terms_of_emp.shape[0]  # nombre d'observations
    assert df.shape[1] == 1  # 1 feature en sortie
    assert (df.index == terms_of_emp.index).all() == True  # conservation des index avant et après le preprocessing
    assert "Terms_of_emp" in df.columns.values  # présence EMPLOYEE_COUNT
    assert (~df.isna()).all().all()==True  # pas de données manquantes
    
    return df


# ===========
# eco_sect_fe
# ===========
def eco_sect_fe(eco_sect: pd.Series) -> pd.DataFrame:
    """eco_sect_fe : traitement de la feature eco_sect

    - eco_sect est fourni sous la forme d'un code en 3 lettres 
    - maintien de eco_sect (category) 
    - les données manquantes (rares) sont imputées par la catégorie majoritaire de Eco_sect (BKU - revoir)

    Args:
        eco_sect (Series): Colonne Eco_sect.

    Returns:
        Dataframe contenant Eco_sect
    """
    
    # Copie de la feature Eco_sect
    # ----------------------------
    df = eco_sect
    
    # Imputation des données manquantes (BKU : à retravailler)
    # ---------------------------------
    categorie_majoritaire = df.Eco_sect.value_counts().index[0]
    df.loc[df.Eco_sect.isna(), "Eco_sect"] = categorie_majoritaire    
    
    # Typage
    # ------
    df.Eco_sect = df.Eco_sect.astype("category")
    
    # Validation des sorties
    # ----------------------
    assert df.shape[0] == eco_sect.shape[0]  # nombre d'observations
    assert df.shape[1] == 1  # 1 feature en sortie
    assert (df.index == eco_sect.index).all() == True  # conservation des index avant et après le preprocessing
    assert "Eco_sect" in df.columns.values  # présence Eco_sect
    assert (~df.isna()).all().all()==True  # pas de données manquantes
    
    return df


# ===================
# work_description_fe
# ===================
def work_description_fe(work_description: pd.Series) -> pd.DataFrame:
    """work_description_fe : traitement de la feature work_description

    - work_description sous la forme abxy 
        . où a est le niveau N1 (entier de 1 à 6, sans ordonnancement ou hiérarchie)
        . où ab est le niveau N2 (entier, sans ordonnancement ou hiérarchie)
        . où xy est une sous-codification alphanumérique du niveau N2 (2 digis, sans ordonnancement ou hiérarchie)
    - création de WD_N1_ (category) : 1er niveau N1 (a)
    - création de WD_N2_ (category) : 2ème niveau N2 (ab)
    - maintien de work_description (category) : codification complète (abxy)

    Args:
        work_description (Series): Colonne work_description.

    Returns:
        Dataframe contenant WD_N1_, WD_N2_, work_description
    """
    
    # Copie de la feature
    # -------------------
    df = work_description    

    # Extraction des niveaux x et y
    # -----------------------------

    # pattern regex pour CSP_x_y
    pattern = r"(\d)(\d)(\d+)"

    # Extraction de x et y
    extracted_values = df.work_description.str.extract(pattern, expand=True)
    extracted_values.columns = ["a", "b", "xy"]  # Renommer les colonnes

    # Création des features WD_N1_ et WD_N2_
    # ---------------------------------------
    df["WD_N1_"] = extracted_values.a
    df["WD_N2_"] = extracted_values.a + extracted_values.b

    # Typage
    # ------
    df["WD_N1_"] = df["WD_N1_"].astype("category")
    df["WD_N2_"] = df["WD_N2_"].astype("category")
    df["work_description"] = df["work_description"].astype("category")
    
    

    # Validation des sorties
    # ----------------------
    assert df.shape[0] == work_description.shape[0]  # nombre d'observations
    assert df.shape[1] == 3  # 3 columns
    assert (df.index != work_description.index).sum() == 0  # conservation des index
    assert "WD_N1_" in df.columns.values  # présence WD_N1_
    assert "WD_N2_" in df.columns.values  # présence WD_N2_
    assert "work_description" in df.columns.values  # présence work_description
    assert (~df.isna()).all().all()==True  # pas de données manquantes  
    
    return df


# ==========
# job_dep_fe
# ==========
def job_dep_fe(job_dep: pd.Series) -> pd.DataFrame:
    """job_dep_fe : traitement de la feature job_dep

    - maintien de job_dep (category) : département du job
    - création de JOB_REGION_ (category) : code région du job (issu du fichier regions.csv)
    - les données manquantes (rares) sont imputées par la catégorie majoritaire de Job_dep (BKU - revoir)

    Args:
        job_dep (Series): Colonne Job_dep.

    Returns:
        Dataframe contenant job_dep / JOB_REGION_
    """
    
    # Copie de la feature job_dep
    # ---------------------------
    df = job_dep
    
    # Imputation des données manquantes (BKU : à retravailler)
    # ---------------------------------
    categorie_majoritaire = df.Job_dep.value_counts().index[0]
    df.loc[df.Job_dep.isna(), "Job_dep"] = categorie_majoritaire
    
    
    # ===============
    # Feature REGION_
    # ===============

    # Collecte des codes régions associés aux départements
    # ----------------------

    # ouverture du dataset "departments.csv" (contenant le code REGION associé aux départements)
    departement_dataset = pd.read_table(f"{cst.DATA_DIR}/departments.csv", sep=",", dtype=object)

    # sélection des colonnes REG et dep
    departement_dataset = departement_dataset[["REG", "dep"]]

    # vérification : pas de redondance dans les départements
    assert departement_dataset["dep"].unique().size == departement_dataset.shape[0]

    # jointure à gauche sur dataframe df (pivot=job_dep/dep)
    # Utiliser pd.join plutôt que pd.merge pour conserver
    # les index de la table initiale
    # ATTENTION : merge perd les index initiaux
    df = df.join(
        other=departement_dataset.set_index("dep"), on="Job_dep", how="left"
    )

    # renommage de la colonne REG en REGION_
    df = df.rename(columns={"REG": "JOB_REGION_"})

    # Typage
    # ------
    df["Job_dep"] = df["Job_dep"].astype("category")
    df["JOB_REGION_"] = df["JOB_REGION_"].astype("category")
    
    # Validation des sorties
    # ----------------------
    assert df.shape[0] == job_dep.shape[0]  # nombre d'observations
    assert df.shape[1] == 2  # 2 columns
    assert (df.index != job_dep.index).sum() == 0  # conservation des index
    assert "Job_dep" in df.columns.values  # présence WD_N1_
    assert "JOB_REGION_" in df.columns.values  # présence WD_N2_
    assert (~df.isna()).all().all()==True  # pas de données manquantes    
    
    return df

# ================
# working_hours_fe
# ================
def working_hours_fe(working_hours: pd.Series) -> pd.DataFrame:
    """working_hours_fe : traitement de la feature working_hours

    - Maintien de working_hours (float)
    - les données manquantes (rares) sont imputées par la valeur médiane (BKU - revoir)

    Args:
        working_hours (Series): Colonne Working_hours.

    Returns:
        Dataframe contenant Working_hours
    """
    
    # Copie de la feature WORK_CONDITION
    # ----------------------------------
    df = working_hours
    
    # Imputation des données manquantes (BKU : à retravailler)
    # ---------------------------------
    mediane = df.Working_hours.median(skipna=True)
    df.loc[df.Working_hours.isna(), "Working_hours"] = mediane
    
    # Typage
    # ------
    df.Working_hours = df.Working_hours.astype("float")
    
    # Validation des sorties
    # ----------------------
    assert df.shape[0] == working_hours.shape[0]  # nombre d'observations
    assert df.shape[1] == 1  # 1 feature en sortie
    assert (df.index == working_hours.index).all() == True  # conservation des index avant et après le preprocessing
    assert "Working_hours" in df.columns.values  # présence Working_hours
    assert (~df.isna()).all().all()==True  # pas de données manquantes
    
            
    return df

# ================
# work_condition_fe
# ================
def work_condition_fe(work_condition: pd.Series) -> pd.DataFrame:
    """work_condition_fe : traitement de la feature Work_condition

    - WORK_CONDITION est fourni sous la forme d'un code à 1 lettre 
    - maintien de WORK_CONDITION (category) 
    - les données manquantes (rares) sont imputées par la catégorie majoritaire de WORK_CONDITION (BKU : à revoir)

    Args:
        work_condition (Series): Colonne WORK_CONDITION.

    Returns:
        Dataframe contenant work_condition
    """
    
    # Copie de la feature WORK_CONDITION
    # ----------------------------
    df = work_condition
    
    # Imputation des données manquantes (BKU : à retravailler)
    # ---------------------------------
    categorie_majoritaire = df.WORK_CONDITION.value_counts().index[0]
    df.loc[df.WORK_CONDITION.isna(), "WORK_CONDITION"] = categorie_majoritaire
    
    # Typage
    # ------
    df.WORK_CONDITION = df.WORK_CONDITION.astype("category")
    
    # Validation des sorties
    # ----------------------
    assert df.shape[0] == work_condition.shape[0]  # nombre d'observations
    assert df.shape[1] == 1  # 1 feature en sortie
    assert (df.index == work_condition.index).all() == True  # conservation des index avant et après le preprocessing
    assert "WORK_CONDITION" in df.columns.values  # présence WORK_CONDITION
    assert (~df.isna()).all().all()==True  # pas de données manquantes   
     
    return df

# ============
# emolument_fe
# ============
def emolument_fe(emolument: pd.Series) -> pd.DataFrame:
    """emolument_fe : traitement de la feature emolument

    - Maintien de working_hours (float)
    - les données manquantes (rares) sont imputées par la valeur médiane (BKU - revoir)

    Args:
        emolument (Series): Colonne EMOLUMENT.

    Returns:
        Dataframe contenant emolument
    """
    
    # Copie de la feature EMOLUMENT
    # -----------------------------
    df = emolument
    
    # Imputation des données manquantes (BKU : à retravailler)
    # ---------------------------------
    mediane = df.EMOLUMENT.median(skipna=True)
    df.loc[df.EMOLUMENT.isna(), "EMOLUMENT"] = mediane
    
    # Typage
    # ------
    df.EMOLUMENT = df.EMOLUMENT.astype("float")
    
    # Validation des sorties
    # ----------------------
    assert df.shape[0] == emolument.shape[0]  # nombre d'observations
    assert df.shape[1] == 1  # 1 feature en sortie
    assert (df.index == emolument.index).all() == True  # conservation des index avant et après le preprocessing
    assert "EMOLUMENT" in df.columns.values  # présence EMOLUMENT
    assert (~df.isna()).all().all()==True  # pas de données manquantes
    
    
    return df


