import pandas as pd
import numpy as np
import datetime
import os
import toolbox_constantes as cst


# *******************
# FEATURE ENGINEERING
# *******************

# =============
# insee_code_fe
# =============
def insee_code_fe(insee_code):
    """insee_code_fe : traitement de la feature insee_code

    - création de DEPARTEMENT_ (category) : 2 premiers digits de insee_code
    - création de REGIONS_ (category) : code région du fichier regions.csv
    - création de TOWN_TYPE_ (entier) : type de commune (variable ordinale)
    - création de INHABITANTS_ (entier) : Nombre d'habitants de la commune
    - création de DISTANCE_GRANDE_VILLE_ (entier) : Distance à la préfecture de département 
    - suppression de la colonne insee_code

    Args:
        insee_code (Serie): Colonne insee_code.

    Returns:
        Dataframe contenant la colonne DEPARTEMENT_ / REGIONS_ / TOWNTYPE_ / INHABITANTS_ / DISTANCE_GRANDE_VILLE_
    """

    # ===================
    # Feature DEPARTEMENT_
    # ===================

    # Extration des 2 premiers caractères de insee code
    df = insee_code.to_frame()
    df["DEPARTEMENT_"] = [s[0:2] for s in insee_code]

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

    # jointure à gauche sur dataframe df (pivot=DEPARTEMENT_/dep)
    # Utiliser pd.join plutôt que pd.merge pour conserver
    # les index de la table initiale
    # ATTENTION : merge perd les index initiaux
    df = df.join(
        other=departement_dataset.set_index("dep"), on="DEPARTEMENT_", how="left"
    )

    # renommage de la colonne REG en REGION_
    df = df.rename(columns={"REG": "REGION_"})

    # Typage
    # ------
    df["DEPARTEMENT_"] = df["DEPARTEMENT_"].astype("category")
    df["REGION_"] = df["REGION_"].astype("category")
    
    # ==================
    # Feature TOWN_TYPE_ : variable ordinale (int64)
    # ==================

    # Collecte des types de communes
    # ----------------------

    # ouverture du dataset "city_adm.csv" (toutes les communes avec type de commune et département)
    city_adm_dataset = pd.read_table(f"{cst.DATA_DIR}/city_adm.csv", sep=",", dtype=object)

    # sélection des colonnes town_type et dep
    city_adm_dataset = city_adm_dataset[["insee_code", "town_type", "dep"]]

    # vérification : pas de redondance dans les codes INSEE
    assert city_adm_dataset["insee_code"].unique().size == city_adm_dataset.shape[0]

    # jointure à gauche sur dataframe df (pivot=DEPARTEMENT_/dep)
    # Utiliser pd.join plutôt que pd.merge pour conserver
    # les index de la table initiale
    # ATTENTION : merge perd les index initiaux
    df = df.join(
        other=city_adm_dataset[["insee_code", "town_type"]].set_index("insee_code"), 
        on="insee_code", 
        how="left"
    )

    # renommage de la colonne REG en REGION_
    df = df.rename(columns={"town_type": "TOWN_TYPE_"})
         
    # mapping des types de communes avec une valeur décroissante
    # --------------------------------------------------------
    map_city = {
        "Capitale d'état": 1,
        "Préfecture de région": 2,
        "Préfecture": 3,
        "Sous-préfecture": 4,
        "Chef-lieu canton": 5,
        "Commune simple": 6
    }
    
    df.TOWN_TYPE_ = df.TOWN_TYPE_.map(map_city)

    # Typage
    # ------
    df["TOWN_TYPE_"] = df["TOWN_TYPE_"].astype("int64")

    # ====================
    # Feature INHABITANTS_
    # ====================
    
    # Collecte des types de communes
    # ----------------------

    # ouverture du dataset "departments.csv"
    city_pop_dataset = pd.read_table(f"{cst.DATA_DIR}/city_pop.csv", sep=",", dtype=object)
    
    # vérification : pas de redondance dans les codes INSEE
    assert city_pop_dataset["insee_code"].unique().size == city_pop_dataset.shape[0]

    # jointure à gauche sur dataframe df (pivot=DEPARTEMENT_/dep)
    # Utiliser pd.join plutôt que pd.merge pour conserver
    # les index de la table initiale
    # ATTENTION : merge perd les index initiaux
    df = df.join(
        other=city_pop_dataset.set_index("insee_code"), 
        on="insee_code", 
        how="left"
    )

    # renommage de la colonne REG en REGION_
    df = df.rename(columns={"Inhabitants": "INHABITANTS_"})
     
    # Typage
    # ------
    df["INHABITANTS_"] = df["INHABITANTS_"].astype("int64")
    
    # ==============================
    # Feature DISTANCE_GRANDE_VILLE_
    # ==============================
    
    
    # Collecte des coordonnées X et Y de toutes les communes
    # ------------------------------------------------------

    # ouverture du dataset "departments.csv"
    city_loc_dataset = pd.read_table(f"{cst.DATA_DIR}/city_loc.csv", sep=",")

    # sélection des coordonnées Lambert 93 (X et Y)
    city_loc_dataset = city_loc_dataset[["insee_code", "X", "Y"]]

    # vérification : pas de redondance dans les codes INSEE
    assert city_loc_dataset["insee_code"].unique().size == city_loc_dataset.shape[0]
    
    # jointure à gauche sur dataframe df (pivot=DEPARTEMENT_/dep) : ajout des colonnes X et Y
    df = df.join(
        other=city_loc_dataset.set_index("insee_code"), 
        on="insee_code", 
        how="left"
    )

    # Collecte de la capitale dans city_admin
    # ---------------------------------------

    # Coordonnée de la CAPITALE (1er arrondissement)
    capitale_insee_codes = city_adm_dataset.loc[city_adm_dataset["town_type"]=="Capitale d'état", "insee_code"] # retourne les insee_codes de tous les arrondisseemnts
    capitale_w_X_Y = city_loc_dataset.loc[city_loc_dataset["insee_code"]==capitale_insee_codes.values[0]] # on choisit le 1er arrondissement de Paris

    # Table des préfectures de région avec coordonnées X et Y
    # -------------------------------------------------------
    
    # Sélection des seules préfectures de région (TOWN_TYPE_=2) dans city_admn
    prefectures_region_insee_codes = city_adm_dataset.loc[city_adm_dataset["town_type"]=="Préfecture de région"]
    # !!! BKU : hors Ile de france (region 11, "Capitale")
    # !!! BKU : les villes avec plusieurs arrondissements (Lyon, marseille) apparaissent plusieurs fois en tant que préfecture de région
    # 21 régions hors Ile-de-France / Marseille = 16 arrondissement / Lyon = 9 arrondissements

    # agreagation des coordonnées lambert 93 (X, Y) pour chaque préfecture de région collectée
    prefectures_region_w_X_Y = prefectures_region_insee_codes.join(
        other=city_loc_dataset.set_index("insee_code"), 
        on="insee_code", 
        how="left"
    )

    # Table des préfectures de département avec coordonnées X et Y
    # ------------------------------------------------------------
    
    # Sélection des seules préfectures de région (TOWN_TYPE_=3) dans city_admn
    prefectures_departement_insee_codes = city_adm_dataset.loc[city_adm_dataset["town_type"]=="Préfecture"]
    # !!! BKU : hors Ile de france ("Capitale")
    # !!! BKU : hors préfecture de région ("Département de région")
    # 74 entrées : 96 départements métropolitains - 22 avec préfecture de région
    
    # agreagation des coordonnées lambert 93 pour chaque préfecture collectée
    prefectures_departement_w_X_Y = prefectures_departement_insee_codes.join(
        other=city_loc_dataset.set_index("insee_code"), 
        on="insee_code", 
        how="left"
    )
    
    # Distance à la grande ville la plus proche : Lambert 93 (distance euclidienne)
    # -----------------------------------------

    # Fonction de Calcul de distance d'une commune à la grande ville la plus proche
    def distance_grande_ville(row_city) -> int:
        
        distance = 0 
        
        # Capitale
        capitale = capitale_w_X_Y 
        
        # Préfecture de région associée à la commune
        prefecture_region = prefectures_region_w_X_Y.loc[prefectures_region_w_X_Y["dep"]==row_city.DEPARTEMENT_]
        
        # Préfecture de département associée à la commune
        prefecture_departement = prefectures_departement_w_X_Y.loc[prefectures_departement_w_X_Y["dep"]==row_city.DEPARTEMENT_]
        
        # choix de la grande ville
        grande_ville = capitale # par défaut, la grande ville la plus proche est la captiale
        if (prefecture_region.shape[0]>0): grande_ville=prefecture_region # sinon c'est la préfecture de région (sauf région Ile-de-France)
        if (prefecture_departement.shape[0]>0): grande_ville=prefecture_departement # sinon c'est la préfecture de département
        
        # Distance
        distance = np.sqrt(np.square(row_city.X - grande_ville.X)  + np.square( row_city.Y - grande_ville.Y ))
        
        return int(distance.values[0])

    temp = df.apply(
        distance_grande_ville, 
        axis=1
    )
    
    df["DISTANCE_GRANDE_VILLE_"] = df.apply(
        distance_grande_ville, 
        axis=1
    )


    # ======================
    # Suppression insee_code
    # ======================

    # suppression des features d'origine (insee_code, X commune, Y commune) 
    # ----------------------------------
    df = df.drop(columns=["insee_code"])
    df = df.drop(columns=["X"])
    df = df.drop(columns=["Y"])


    # Validation des sorties
    # ----------------------
    assert df.shape[0] == insee_code.shape[0]  # nombre d'observations
    assert df.shape[1] == 5
    assert (df.index != insee_code.index).sum() == 0  # conservation des index
    assert "DEPARTEMENT_" in df.columns.values  # présence DEPARTEMENT_
    assert "REGION_" in df.columns.values  # présence REGION_
    assert "TOWN_TYPE_" in df.columns.values  # présence TOWN_TYPE_
    assert "INHABITANTS_" in df.columns.values  # présence TOWN_TYPE_
    assert "DISTANCE_GRANDE_VILLE_" in df.columns.values  # présence TOWN_TYPE_
    assert "insee_code" not in df.columns.values  # absence insee_code
    assert "X" not in df.columns.values  # absence X
    assert "Y" not in df.columns.values  # absence Y
    assert (~df.isna()).all().all()==True  # pas de données manquantes
    
    return df


# ================
# OCCUPATION_42_fe
# ================
def OCCUPATION_42_fe(OCCUPATION_42):
    """traitement de la feature OCCUPATION_42

    - OCCUPATION_42 sous la forme CSP_x_y
    - création de CSP_N1_ (category) : 1er niveau CSP (x)
    - création de CSP_N2_ (category) : 2ème niveau CSP (xy)
    - suppression de la colonne OCCUPATION_42

    Args:
        OCCUPATION_42 (Serie): Colonne OCCUPATION_42.

    Returns:
        Dataframe contenant les colonnes CSP_N1_ et CSP_N2_
    """

    # Extraction des niveaux x et y
    # -----------------------------

    # pattern regex pour CSP_x_y
    pattern = r"csp_(\d+)_(\d+)"

    # Extraction de x et y
    extracted_values = OCCUPATION_42.str.extract(pattern, expand=True)
    extracted_values.columns = ["x", "y"]  # Renommer les colonnes

    # Création des features CSP_N1_ et CSP_N2_
    # ----------------------------------------
    df = OCCUPATION_42.to_frame()
    df["CSP_N1_"] = extracted_values.x
    df["CSP_N2_"] = extracted_values.x + extracted_values.y

    # suppression de la feature d'origine
    # -----------------------------------
    df = df.drop(columns=["OCCUPATION_42"])

    # Typage
    # ------
    df["CSP_N1_"] = df["CSP_N1_"].astype("category")
    df["CSP_N2_"] = df["CSP_N2_"].astype("category")

    # Validation des sorties
    # ----------------------
    assert df.shape[0] == OCCUPATION_42.shape[0]  # nombre d'observations
    assert df.shape[1] == 2  # 2 columns
    assert (df.index != OCCUPATION_42.index).sum() == 0  # conservation des index
    assert "CSP_N1_" in df.columns.values  # présence CSP_N1_
    assert "CSP_N2_" in df.columns.values  # présence CSP_N2_
    assert (~df.isna()).all().all()==True  # pas de données manquantes

    return df


# ================
# ACTIVITY_TYPE_fe
# ================
def ACTIVITY_TYPE_fe(ACTIVITY_TYPE):
    """traitement de la feature ACTIVITY_TYPE

    - ACTIVITY_TYPE sous la forme TYPEa|b
    - création de ACTIVITY_L1_ (category) : activité (1=actif / 2=inactif)
    - création de ACTIVITY_L2_ (category) : type d'activité/inactivité
    - suppression de la colonne ACTIVITY_TYPE

    Args:
        ACTIVITY_TYPE (Serie): Colonne ACTIVITY_TYPE.

    Returns:
        Dataframe contenant les colonnes ACTIVITY_L1_ et ACTIVITY_L2_
    """

    # Extraction des niveaux x et y
    # -----------------------------

    # pattern regex pour CSP_x_y
    pattern = r"TYPE(\d+)\|(\d+)"

    # Extraction de x et y
    extracted_values = ACTIVITY_TYPE.str.extract(pattern, expand=True)
    extracted_values.columns = ["a", "b"]  # Renommer les colonnes

    # Création des features ACTIVITY_L1_ et ACTIVITY_L2_
    # ----------------------------------------
    df = ACTIVITY_TYPE.to_frame()
    df["ACTIVITY_L1_"] = extracted_values.a
    df["ACTIVITY_L2_"] = extracted_values.a + extracted_values.b

    # suppression de la feature d'origine
    # -----------------------------------
    df = df.drop(columns=["ACTIVITY_TYPE"])

    # Typage
    # ------
    df["ACTIVITY_L1_"] = df["ACTIVITY_L1_"].astype("category")
    df["ACTIVITY_L2_"] = df["ACTIVITY_L2_"].astype("category")

    # Validation des sorties
    # ----------------------
    assert df.shape[0] == ACTIVITY_TYPE.shape[0]  # nombre d'observations
    assert df.shape[1] == 2  # 2 columns
    assert (df.index != ACTIVITY_TYPE.index).sum() == 0  # conservation des index
    assert "ACTIVITY_L1_" in df.columns.values  # présence ACTIVITY_L1_
    assert "ACTIVITY_L2_" in df.columns.values  # présence ACTIVITY_L2_
    assert (~df.isna()).all().all()==True  # pas de données manquantes

    return df


# ============
# household_fe
# ============
def household_fe(household):
    """traitement de la feature household

    - household sous la forme TYPMRa-b
    - création de HOUSEHOLD_L1_ (category) : configuration du foyer
    - création de HOUSEHOLD_L2_ (category) : type de la configuration
    - suppression de la colonne household


    Args:
        household (Serie): Colonne household.

    Returns:
        Dataframe contenant les colonnes HOUSEHOLD_L1_ et HOUSEHOLD_L2_
    """

    # Extraction des niveaux x et y
    # -----------------------------

    # pattern regex pour CSP_x_y
    pattern = r"TYPMR(\d+)-(\d+)"

    # Extraction de x et y
    extracted_values = household.str.extract(pattern, expand=True)
    extracted_values.columns = ["a", "b"]  # Renommer les colonnes

    # Création des features HOUSEHOLD_L1_ et HOUSEHOLD_L2_
    # ----------------------------------------
    df = household.to_frame()
    df["HOUSEHOLD_L1_"] = extracted_values.a
    df["HOUSEHOLD_L2_"] = extracted_values.a + extracted_values.b

    # suppression de la feature d'origine
    # -----------------------------------
    df = df.drop(columns=["household"])

    # Typage
    # ------
    df["HOUSEHOLD_L1_"] = df["HOUSEHOLD_L1_"].astype("category")
    df["HOUSEHOLD_L2_"] = df["HOUSEHOLD_L2_"].astype("category")

    # Validation des sorties
    # ----------------------
    assert df.shape[0] == household.shape[0]  # nombre d'observations
    assert df.shape[1] == 2  # 2 columns
    assert (df.index != household.index).sum() == 0  # conservation des index
    assert "HOUSEHOLD_L1_" in df.columns.values  # présence HOUSEHOLD_L1_
    assert "HOUSEHOLD_L2_" in df.columns.values  # présence HOUSEHOLD_L2_
    assert (~df.isna()).all().all()==True  # pas de données manquantes

    return df


# =================
# Highest_degree_fe
# =================
def Highest_degree_fe(Highest_degree):
    """Traitement de la feature Highest_degree

    - Highest_degree est sous la forme edu.a[.b]
    - création de DEGREE_ (int64) : représentation numérique du diplôme (0 si pas de diplôme, b sinon)
    - suppression de la colonne Highest_degree

    Args:
        Highest_degree (Serie): Colonne Highest_degree.

    Returns:
        Dataframe contenant la colonne DEGREE_
    """

    # Normaliser le format des modalités (i.e. ajouter ".0" à "edu.1/2/3")
    # ----------------------------------
    df = Highest_degree.to_frame()
    df["DEGREE"] = [
        s + ".0" if len(s) == len("edu.x") else s for s in df.Highest_degree
    ]

    # Extraction des niveaux x et y
    # -----------------------------

    # pattern regex pour edu.a.b
    pattern = r"edu\.(\d+)\.(\d+)"

    # Extraction de x et y
    extracted_values = df["DEGREE"].str.extract(pattern, expand=True)
    extracted_values.columns = ["a", "b"]  # Renommer les colonnes

    # Création de la feature DEGREE_
    # ------------------------------
    df["DEGREE_"] = extracted_values.b

    # suppression de la feature d'origine
    # -----------------------------------
    df = df.drop(columns=["Highest_degree"])
    df = df.drop(columns=["DEGREE"])

    # Typage (numeric)
    # ------
    df["DEGREE_"] = df["DEGREE_"].astype("int64")

    # Validation des sorties
    # ----------------------
    assert df.shape[0] == Highest_degree.shape[0]  # nombre d'observations
    assert df.shape[1] == 1  # 1 column
    assert (df.index != Highest_degree.index).sum() == 0  # conservation des index
    assert "DEGREE_" in df.columns.values  # présence DEGREE_
    assert (~df.isna()).all().all()==True  # pas de données manquantes

    return df


# ======
# sex_fe
# ======
def sex_fe(sex):
    """Traitement de la feature sex

    - sex est Male/Female (object)
    - typage de la feature sex en category

    Args:
        sex (Serie): Colonne sex.

    Returns:
        Dataframe contenant la colonne sex (category)
    """

    # Typage (category)
    # ------
    df = sex.to_frame()
    df["sex"] = df["sex"].astype("category")

    # Validation des sorties
    # ----------------------
    assert df.shape[0] == sex.shape[0]  # nombre d'observations
    assert df.shape[1] == 1  # 1 column
    assert (df.index != sex.index).sum() == 0  # conservation des index
    assert "sex" in df.columns.values  # présence sex
    assert (~df.isna()).all().all()==True  # pas de données manquantes

    return df


# ======
# Id_fe
# ======
def Id_fe(Id):
    """Traitement de la feature Id

    - Suppression de la colonne Id

    Args:
        Id (Serie): Colonne Id

    Returns:
        Dataframe vide
    """

    # Data frame vide
    # ---------------
    df = pd.Dataframe()

    # Validation des sorties
    # ----------------------
    assert df.shape[1] == 0  # 0 column

    return df

# =======
# Club_fe
# =======
def Club_fe(Club: pd.Series) -> pd.DataFrame:
    """Club_fe : traitement de la feature Club (dataset Sport)

    - création de SPORTIF_ (booleen) : TRUE si un club est renseigné, FALSE si NA
    - création de HANDICAP_ (categoriel) : selon la catégorie de club
    - suppression de la colonne Club

    Args:
        Club (Serie): Colonne Club.

    Returns:
        Dataframe contenant les variables extraites
    """

    df = Club.to_frame()

    # ================
    # Feature SPORTIF_
    # ================

    # True si un club est rensigné 
    df["SPORTIF_"] = df.Club.isna()==False

    # =================
    # Feature HANDICAP_
    # =================

    # Collecte des catégories associées aux clubs
    # -------------------------------------------

    # ouverture du dataset "departments.csv"
    code_club_dataset = pd.read_table(f"{cst.DATA_DIR}/code_Club.csv", sep=",", dtype=object)

    # sélection des colonnes Code et Categorie
    code_club_dataset = code_club_dataset[["Code", "Categorie"]]

    # vérification : pas de redondance dans les codes Fédération
    assert code_club_dataset["Code"].unique().size == code_club_dataset.shape[0]

    # jointure à gauche sur dataframe df (pivot=Club/Code)
    df = df.join(
        other=code_club_dataset.set_index("Code"), 
        on="Club", 
        how="left"
    )

    # renommage de la colonne REG en REGION_
    # df = df.rename(columns={"Cat": "REGION_"})
    df['HANDICAP_'] = df['Categorie'].apply(lambda x: 'HANDICAP' if x == "6" else ('SANS_HANDICAP' if pd.notna(x) else 'NON_RENSEIGNE'))
    # df['STATUT_'] = df['Categorie'].apply(lambda x: 'MODALITE_AVEC' if x == 6 else ('MODALITE_SANS' if pd.notna(x) else 'MODALITE_NON_RENSEIGNE'))

    # Typage
    # ------
    df["SPORTIF_"] = df["SPORTIF_"].astype("bool")
    df["HANDICAP_"] = df["HANDICAP_"].astype("category")
    
    
    # ================
    # Suppression Club
    # ================

    # suppression de Club et Categorie
    # -------------------
    df = df.drop(columns=["Club", "Categorie"])


    # Validation des sorties
    # ----------------------
    assert df.shape[0] == Club.shape[0]  # nombre d'observations
    assert df.shape[1] == 2  # 2 columns
    assert (df.index != Club.index).sum() == 0  # conservation des index
    assert "SPORTIF_" in df.columns.values
    assert "HANDICAP_" in df.columns.values  # présence REGION_
    assert "Club" not in df.columns.values  # absence insee_code
    assert "Categorie" not in df.columns.values  # absence insee_code
    assert (~df.isna()).all().all()==True  # pas de données manquantes
    
    return df


# ===============
# Emp_contract_fe
# ===============
def Emp_contract_fe(Emp_contract_df: pd.DataFrame) -> pd.DataFrame:
    """Emp_contract_fe : traitement de la feature Emp_contract (dataset Emp_contract)

    - création de CONTRAT_ (category) : selon la catégorie de Emp_contract (CDD / CDI / ENTREPRENEUR / SANS_CONTRAT)
    - suppression de la colonne Emp_contract

    Args:
        Emp_contract (Serie): Colonnes Emp_contract (à exploiter) et ACTIVITY_TYPE (pour la gestion des données manquantes)

    Returns:
        Dataframe contenant les variables extraites
    """

    df = Emp_contract_df
    
    # ===============================
    # Imputation des données manquantes de Emp_contract 
    # ===============================
    
    # Définissez une fonction pour déterminer la valeur de la colonne EMP_NEW_
    def determiner_contrat(row):
        
        # Emp_contract est connu
        # ----------------------
        if pd.notna(row['Emp_contract']):
            
            # 1.6 = CDI
            if (row.Emp_contract=="COND.1.6"):
                contract="CONTRAT_DUREE_DETERMINEE"
            # 1.x = contrat à durée indéterminée
            elif (row.Emp_contract.startswith("COND.1.")):
                contract="CONTRAT_DUREE_INDETERMINEE"
            # 2.2 et 2.3 = entrepreneur
            elif (row.Emp_contract=="COND.2.1" or row.Emp_contract=="COND.2.2" ):
                contract="ENTREPRENEUR"
            else:
                contract="SANS_CONTRAT"
            
            return contract
        
        # Emp_contract est NA
        # -------------------
        
        # Activity_type indique "actif avec un emploi" (TYPE1|1) => on extrapole DUREE_INDETERMINEE
        elif row['ACTIVITY_TYPE'] == 'TYPE1|1':
            return 'CONTRAT_DUREE_DETERMINEE'
        # sinon SANS_CONTRAT
        else:
            return 'SANS_CONTRAT'

    # Appliquez la fonction pour créer la nouvelle colonne EMP_NEW_
    df['CONTRAT_'] = df.apply(determiner_contrat, axis=1)
    
    # Typage
    # ------
    df["CONTRAT_"] = df["CONTRAT_"].astype("category")


    # ================
    # Suppression Emp_contrat et ACTIVITY_TYPE
    # ================

    df = df.drop(columns=["Emp_contract", "ACTIVITY_TYPE"])


    # Validation des sorties
    # ----------------------
    assert df.shape[0] == Emp_contract_df.shape[0]  # nombre d'observations
    assert df.shape[1] == 1  # 2 columns
    assert (df.index != Emp_contract_df.index).sum() == 0  # conservation des index
    assert "CONTRAT_" in df.columns.values
    assert "Emp_contract" not in df.columns.values  # absence Emp_contract
    assert "ACTIVITY_TYPE" not in df.columns.values  # absence ACTIVTY_TYPE
    assert (~df.isna()).all().all()==True  # pas de données manquantes
    
    return df

