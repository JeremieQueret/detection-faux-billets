import os
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Détermine le chemin absolu du fichier actuel
current_file = os.path.abspath(__file__)

# Détermine le répertoire de base du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), ".."))

# Définir le chemin du sous-repertoire 'processed'
processed_data_dir = os.path.join(project_root, "data", "processed")

# Créer le répertoire 'processed' si il n'existe pas
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)
    print(f"[INFO] Création du répertoire : {processed_data_dir}")


class StaticLinearRegressionImputer(BaseEstimator, TransformerMixin):
    """
    Impute les valeurs manquantes dans une colonne cible en utilisant une régression linéaire multiple.

    Cette classe est une implémentation personnalisée de scikit-learn qui permet d'imputer les valeurs
    manquantes d'une colonne cible (`target_column`) à l'aide d'un modèle de régression linéaire
    entraîné sur les colonnes prédictrices spécifiées (`predictor_columns`).

    Attributs :
    ----------
    target_column : str
        Nom de la colonne cible contenant des valeurs manquantes à imputer.

    predictor_columns : list of str, optional
        Liste des noms des colonnes prédictrices utilisées pour entraîner le modèle de régression.

    model : LinearRegression
        Modèle de régression linéaire utilisé pour effectuer les imputations.

    selected_features : list of str
        Liste des colonnes prédictrices sélectionnées après l'entraînement.

    r2_score_ : float
        Coefficient de détermination (R²) du modèle sur les données d'entraînement.

    mae_ : float
        Erreur absolue moyenne (Mean Absolute Error, MAE) sur les données d'entraînement.

    mse_ : float
        Erreur quadratique moyenne (Mean Squared Error, MSE) sur les données d'entraînement.

    rmse_ : float
        Racine carrée de l'erreur quadratique moyenne (Root Mean Squared Error, RMSE) sur les données d'entraînement.

    missing_before_ : int
        Nombre de valeurs manquantes dans la colonne cible avant la transformation.

    missing_after_ : int
        Nombre de valeurs manquantes dans la colonne cible après la transformation.

    Méthodes :
    ---------
    fit(X, y=None)
        Entraîne le modèle de régression linéaire en utilisant les colonnes prédictrices
        pour estimer la colonne cible inversée (1/valeur cible).

    transform(X)
        Impute les valeurs manquantes dans la colonne cible en utilisant les prédictions du modèle.

    """

    def __init__(self, target_column, predictor_columns=None):
        self.target_column = target_column
        self.predictor_columns = predictor_columns
        self.model = LinearRegression()
        self.selected_features = None
        self.r2_score_ = None
        self.mae_ = None
        self.mse_ = None
        self.rmse_ = None
        self.missing_before_ = 0
        self.missing_after_ = 0

    def fit(self, X, y=None):
        if not self.predictor_columns:
            raise ValueError("Les colonnes prédictrices doivent être spécifiées.")

        self.missing_before_ = X[self.target_column].isna().sum()

        # Créer une copie explicite pour éviter SettingWithCopyWarning
        complete_data = X.dropna(subset=[self.target_column]).copy()
        # Transformation inverse de la variable cible
        complete_data["margin_low_inverse"] = 1 / complete_data[self.target_column]

        y_train = complete_data["margin_low_inverse"]
        X_train = complete_data[self.predictor_columns]

        self.selected_features = self.predictor_columns
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_train)
        self.r2_score_ = r2_score(y_train, y_pred)
        self.mae_ = mean_absolute_error(y_train, y_pred)
        self.mse_ = mean_squared_error(y_train, y_pred)
        self.rmse_ = np.sqrt(self.mse_)

        return self

    def transform(self, X):
        X_transformed = X.copy()

        incomplete_data = X_transformed[X_transformed[self.target_column].isna()]
        if not incomplete_data.empty:
            predicted_margin_low_inverse = self.model.predict(
                incomplete_data[self.selected_features]
            )
            X_transformed.loc[incomplete_data.index, self.target_column] = (
                1 / predicted_margin_low_inverse
            )

        self.missing_after_ = X_transformed[self.target_column].isna().sum()

        return X_transformed

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        output_path = os.path.join(processed_data_dir, filename)
        try:
            df.to_csv(output_path, index=False, sep=";", encoding="utf-8")
            print(
                f"[INFO] Données transformées sauvegardées avec succès : {output_path}"
            )
        except Exception as e:
            raise RuntimeError(
                f"[ERREUR] Une erreur s'est produite lors de la sauvegarde du fichier '{output_path }': {e}"
            )


print("[DEBUG] Script chargé")

if __name__ == "__main__":
    print("[DEBUG] Début du script")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    try:

        # Importer la fonction load_data depuis le module load_data.py situé dans le dossier parent
        from load_data import load_data

        # Charger les données
        data = load_data("billets.csv")

        # Instancier l'imputer
        imputer = StaticLinearRegressionImputer(
            target_column="margin_low",
            predictor_columns=[
                "diagonal",
                "height_left",
                "height_right",
                "length",
                "margin_up",
            ],
        )

        # Entrainer l'imputer sur les données
        imputer.fit(data)
        print(
            f"\n[INFO] Score R² sur les données d'entraînement : {imputer.r2_score_:.4f}"
        )
        print(f"[INFO] Erreur Absolue Moyenne (MAE) : {imputer.mae_:.4f}")
        print(f"[INFO] Erreur Quadratique Moyenne (MSE) : {imputer.mse_:.4f}")
        print(
            f"[INFO] Racine de l'Erreur Quadratique Moyenne (RMSE) : {imputer.rmse_:.4f}"
        )
        print(
            f"[INFO] Nombre de valeurs manquantes avant imputation : {imputer.missing_before_}"
        )

        # Transformer les données
        data_imputed = imputer.transform(data)
        print(
            f"[INFO] Nombre de valeurs manquantes après imputation : {imputer.missing_after_}"
        )
        print("\n[INFO] Aperçu des données imputées :")
        print(data_imputed.head())

        # Sauvegarder les données imputées
        imputer.save_processed_data(data_imputed, "billets_imputed.csv")

    except ImportError as e:
        print(
            f"[ERREUR] Erreur d'importation : {e}. Assurez-vous que 'load_data.py' est dans le répertoire parent."
        )
    except FileNotFoundError as e:
        print(f"[ERREUR] Fichier non trouvé : {e}")
    except ValueError as e:
        print(f"[ERREUR] Erreur de valeur : {e}")
    except RuntimeError as e:
        print(f"[ERREUR] Erreur d'exécution : {e}")
    except Exception as e:
        print(f"[ERREUR] Une erreur inattendue s'est produite : {e}")
