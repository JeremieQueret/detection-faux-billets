import os
import pandas as pd

# Détermine le chemin absolu du fichier actuel
current_file = os.path.abspath(__file__)

# Détermine le répertoire de base du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_data(filename: str) -> pd.DataFrame:
    """
    Charge un fichier CSV depuis le répertoire data/raw du projet.
    """

    if not filename.lower().endswith('.csv'):
        raise ValueError(f"Le fichier '{filename}' n'est pas un fichier .csv.")

    data_path = os.path.join(project_root, "data", "raw", filename)

    if not os.path.exists(data_path):
        raw_dir = os.path.join(project_root, "data", "raw")
        available_files = os.listdir(raw_dir) if os.path.exists(raw_dir) else []
        raise FileNotFoundError(
            f"Le fichier '{data_path}' n'existe pas.\n"
            f"Fichiers disponibles dans 'data/raw' : {available_files}"
        )
        raise FileNotFoundError(f"Le fichier '{data_path}' n'existe pas.")

    try:
        df = pd.read_csv(data_path, sep=";", encoding="utf-8")
        print(f"[INFO] Fichier chargé avec succès : {data_path}")
        print("[INFO] Aperçu des données :")
        print(df.head())
        return df
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du fichier '{data_path}' : {e}")


if __name__ == "__main__":
    try:
        df = load_data("billets.csv")
    except Exception as err:
        print(f"[ERREUR] {err}")
