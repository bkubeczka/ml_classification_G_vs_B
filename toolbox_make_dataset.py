import pandas as pd
# import numpy as np
# import datetime
# import os
import toolbox_constantes as cst

# %%

def make_dataset_main(ds_main: pd.DataFrame, 
                       ds_job: pd.DataFrame,
                       ds_emp_contract: pd.DataFrame,
                       ds_sport: pd.DataFrame) -> pd.DataFrame:
    

    df=ds_main
    
    # =============
    # Dataset Sport
    # =============
    
    # agrégatiion du dataset main et sport (pivot Id)
    df = df.join(
        other=ds_sport.set_index("Id"), 
        on="Id", 
        how="left"
    )
    
    # ===========
    # Dataset Job
    # ===========
    
    # agrégation du dataset main et Job (pivot Id)
    # sous-ensemble des individus du jeu de données principal : les données sont intégrées avec un grand nombre de données manquantes
    df = df.join(
        other=ds_job.set_index("Id"), 
        on="Id", 
        how="left"
    )

    # ====================
    # Dataset Emp_contract
    # ====================
    
    # agrégatiion du dataset main et Job (pivot Id)
    df = df.join(
        other=ds_emp_contract.set_index("Id"), 
        on="Id", 
        how="left"
    )

    # vérification de la sortie
    # -------------------------
    assert df.shape[0] == ds_main.shape[0]  # nombre d'observations identique au dataset initial
    assert (df.shape[1] == ds_main.shape[1] + 
                            (ds_sport.shape[1]-1) + 
                            (ds_job.shape[1]-1) + 
                            (ds_emp_contract.shape[1]-1)
            ) # nombre de colonnes = somme des colonnes (sans Id)
    assert (df.Id==ds_main.Id).all()==True  # conservation des index 

    # sport
    assert "Club" in df.columns.values  # présence Club
    assert(ds_sport.shape[0]==(~df.Club.isna()).sum()) # import de toutes les valeurs
    
    # Job
    assert "employer_category" in df.columns.values
    assert "job_category" in df.columns.values
    assert "EMPLOYEE_COUNT" in df.columns.values
    assert "Terms_of_emp" in df.columns.values
    assert "Eco_sect" in df.columns.values
    assert "work_description" in df.columns.values
    assert "WORK_CONDITION" in df.columns.values
    assert "EMOLUMENT" in df.columns.values
    assert(ds_job.shape[0]==(~df.job_category.isna()).sum()) # import de toutes les valeurs
    
    # emp_contract
    assert "Emp_contract" in df.columns.values
    assert(ds_emp_contract.shape[0]==(~df.Emp_contract.isna()).sum()) # import de toutes les valeurs

    return df
