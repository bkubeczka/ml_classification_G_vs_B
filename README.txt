REPERTOIRES

root : code + notebook quarto
data : copie des fichiers de données
models : fichiers sérialisés générés par les étapes du processus à destination des étapes suivantes
logs : traces générées à l'exécution
reports : sortie des fichiers PDF générés par Quarto


ENVIRONNEMENT D'EXECUTION

Le projet a été développé 
- sous Windows
- en python 3.10.5 
- en environnement virtualisé construit selon la procédure suivante :

sous WINDOWS Powershell

cd <répertoire>
python -m venv ./.venv
./.venv/Scripts/activate
pip install -r requirements.txt


EXECUTION du PROJET (2 à 3h)

make_all.bat ou make_all.sh : exécution complète du projet avec génération finale du projet
