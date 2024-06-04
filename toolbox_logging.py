import logging
import datetime
import os
import toolbox_constantes as cst

# ****************
# GESTION des LOGS
# ****************

# *********
# getLogger
# *********
def getLogger(module_name="", logging_level=logging.INFO):
    """retourne un logger sortie standard + fichier de log"""

    # Nom du fichier de logs avec la date et l'heure actuelles
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_file_name = f"{cst.LOGS_DIR}/{module_name}_{current_time}.log"

    # Configuration du logger pour le module "001_make_dataset"
    logger = logging.getLogger(module_name)
    logger.setLevel(logging_level)

    # Ajout du gestionnaire de fichier au logger
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging_level)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # sortie console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    return logger


# ===========
# print_phase
# ===========
def print_phase(logger, label):
    """affichage d'un titre de phase"""

    frame = "=" * len(label)
    logger.info(f"{frame}")
    logger.info(label)
    logger.info(f"{frame}")
    logger.info("")


# =============
# print_section
# =============
def print_section(logger, label, ln=False):
    """affichage d'un titre de section"""

    frame = "-" * len(label)
    # logger.info(f'>> {label}')
    # logger.info(f'   {frame}')
    logger.info(f"{frame}")
    logger.info(label)
    logger.info(f"{frame}")
    if ln == True:
        logger.info("")

