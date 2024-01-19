import configparser
import os
def get_mode_calcul():
    # Lire la configuration depuis le fichier
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Obtenir la valeur de CALCUL_MODE
    calcul_mode = config.get('PARAMETRES', 'CALCUL_MODE')
    return calcul_mode