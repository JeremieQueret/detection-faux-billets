import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    PrecisionRecallDisplay,
)
from sklearn.pipeline import Pipeline

from preprocessing import StaticLinearRegressionImputer
from load_data import load_data


class ModelTrainer:
    """
    Classe pour entraîner et évaluer un modèle de classification de billets.
    """

    def __init__(
        self,
        raw_data_path="billets.csv",
        test_size=0.3,
        random_state=42,
        stratify_target=True,
        predictor_columns=None,
        imputer_target_column="margin_low",
    ):
        """
        Initialise le ModelTrainer.

        Args:
            raw_data_path (str): Chemin vers le fichier de données brutes.
            test_size (float): Proportion des données à utiliser pour l'ensemble de test.
            random_state (int): Graine aléatoire pour la reproductibilité.
            stratify_target (bool): Indique s'il faut stratifier l'échantillonage sur la variable cible.
            predictor_columns (list): Liste des noms de colonnes à utiliser comme features pour l'entraînement du modèle.
                                      La colonne cible de l'imputation ne doit pas être incluse ici.
                                      Si None, toutes les colonnes sauf la cible ('is_genuine') sont utilisées.
            imputer_target_column (str): Nom de la colonne cible pour l'imputation.
        """
        self.now = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # Garder pour info potentielle mais pas pour les noms de fichiers
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.data_dir = os.path.join(self.project_root, "data", "processed")
        self.results_dir = os.path.join(self.project_root, "results")
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        self.logit_metrics_dir = os.path.join(self.results_dir, "logit_metrics")
        self.visu_dir = os.path.join(self.results_dir, "visualizations")
        self.model_dir = os.path.join(self.project_root, "model")
        self.raw_data_path = raw_data_path
        self.test_size = test_size
        self.random_state = random_state
        self.stratify_target = stratify_target
        self.predictor_columns = predictor_columns
        self.imputer_target_column = imputer_target_column
        self.pipeline = None
        self.imputer = None  # Initialisation de imputer
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        for d in [
            self.data_dir,
            self.metrics_dir,
            self.logit_metrics_dir,
            self.visu_dir,
            self.model_dir,
        ]:
            os.makedirs(d, exist_ok=True)

    def load_and_split_data(self):
        """
        Charge les données, sépare les features et la target, et divise les données en ensembles d'entraînement et de test.
        """
        df = load_data(self.raw_data_path)
        self.y = df["is_genuine"].astype(int)
        self.X = df.drop(columns=["is_genuine"])
        if self.predictor_columns:
            # Assurez-vous que la colonne cible de l'imputation est incluse dans X au départ
            if self.imputer_target_column not in self.X.columns:
                raise ValueError(
                    f"La colonne cible de l'imputation '{self.imputer_target_column}' n'est pas présente dans les données."
                )
            # Sélection des colonnes pour l'entraînement du modèle (sans la target de l'imputation)
            columns_to_use = [
                col
                for col in self.predictor_columns
                if col != self.imputer_target_column
            ]
            self.X_for_model = self.X[columns_to_use]
        else:
            self.X_for_model = self.X.drop(
                columns=[self.imputer_target_column], errors="ignore"
            )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y if self.stratify_target else None,
        )

        # Sauvegarde des datasets (utilisation de X original pour l'imputation lors du fit)
        self._save_data(self.X_train, "X_train.csv")
        self._save_data(self.X_test, "X_test.csv")
        self._save_data(self.y_train, "y_train.csv")
        self._save_data(self.y_test, "y_test.csv")
        self._save_data(self.X, "X.csv")
        self._save_data(self.y, "y.csv")

    def _save_data(self, df, filename):
        """
        Méthode interne pour sauvegarder un DataFrame au format CSV.
        """
        df.to_csv(os.path.join(self.data_dir, filename), sep=";", index=False)

    def create_pipeline(self):
        """
        Crée le pipeline de prétraitement et le modèle de classification.
        """
        if self.predictor_columns is None:
            # Utiliser toutes les colonnes sauf 'is_genuine' et la target de l'imputation
            final_predictor_columns = [
                col
                for col in self.X.columns
                if col not in ["is_genuine", self.imputer_target_column]
            ]
        else:
            final_predictor_columns = [
                col
                for col in self.predictor_columns
                if col != self.imputer_target_column
            ]

        imputer = StaticLinearRegressionImputer(
            predictor_columns=final_predictor_columns,
            target_column=self.imputer_target_column,  # Ordre corrigé
        )
        self.pipeline = Pipeline(
            [
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )
        self.imputer = imputer  # Garder une référence à l'imputer pour les métriques

    def train_model(self):
        """
        Entraîne le modèle sur les données d'entraînement.
        """
        if self.pipeline is None or self.X_train is None or self.y_train is None:
            raise RuntimeError(
                "Le pipeline doit être créé et les données chargées avant l'entraînement."
            )
        self.pipeline.fit(self.X_train, self.y_train)

        # Sauvegarde des métriques de l'imputer
        if self.imputer:  # Vérifier si l'imputer a été créé
            imputer_metrics = {
                "intercept": self.imputer.model.intercept_.tolist(),
                "coefficients": dict(
                    zip(
                        self.imputer.selected_features,
                        self.imputer.model.coef_.tolist(),
                    )
                ),
                "r2": self.imputer.r2_score_,
                "mae": self.imputer.mae_,
                "mse": self.imputer.mse_,
                "rmse": self.imputer.rmse_,
            }
            self._save_metrics(
                imputer_metrics, "imputer_metrics.json", self.metrics_dir
            )

    def evaluate_model(self):
        """
        Évalue le modèle sur les ensembles d'entraînement et de test.
        Génère et sauvegarde les métriques, la matrice de confusion et la courbe ROC.
        """
        if self.pipeline is None or self.X_test is None or self.y_test is None:
            raise RuntimeError(
                "Le pipeline doit être entraîné et les données chargées avant l'évaluation."
            )

        train_score = self.pipeline.score(self.X_train, self.y_train)
        test_score = self.pipeline.score(self.X_test, self.y_test)
        logit_scores = {"train_score": train_score, "test_score": test_score}
        self._save_metrics(logit_scores, "logit_scores.json", self.logit_metrics_dir)

        # Rapport de classification
        report = classification_report(
            self.y_test, self.pipeline.predict(self.X_test), output_dict=True
        )
        self._save_metrics(report, "classification_report.json", self.logit_metrics_dir)

        # Matrice de confusion
        cm = confusion_matrix(self.y_test, self.pipeline.predict(self.X_test))
        ConfusionMatrixDisplay(cm).plot()
        plt.title("Matrice de confusion")
        plt.savefig(os.path.join(self.visu_dir, "confusion_matrix.png"))
        plt.close()

        # Courbe ROC
        y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Courbe ROC")
        plt.legend()
        plt.savefig(os.path.join(self.visu_dir, "roc_curve.png"))
        plt.close()

        # Probabilités d'appartenance
        plt.figure()
        plt.hist(
            [y_proba[self.y_test == 0], y_proba[self.y_test == 1]],
            label=["Faux", "Vrais"],
            bins=20,
        )
        plt.title("Probabilités d'appartenance aux classes")
        plt.xlabel("Probabilité")
        plt.ylabel("Nombre de billets")
        plt.legend()
        plt.savefig(os.path.join(self.visu_dir, "probabilites_classes.png"))
        plt.close()

        # Courbe precision-rappel
        PrecisionRecallDisplay.from_predictions(
            self.y_test, y_proba, name="Logistic Regression", plot_chance_level=True
        )
        plt.title("Precision-Recall")
        plt.savefig(os.path.join(self.visu_dir, "precision_recall.png"))
        plt.close()

    def _save_metrics(self, metrics, filename, output_dir):
        """
        Méthode interne pour sauvegarder les métriques au format JSON.
        """
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(metrics, f, indent=4)

    def save_model(self, filename="pipeline_logit"):
        """
        Sauvegarde le pipeline entraîné.

        Args:
            filename (str): Nom de base du fichier de sauvegarde du modèle.
        """
        if self.pipeline is None:
            raise RuntimeError("Le pipeline doit être entraîné avant la sauvegarde.")
        model_path = os.path.join(self.model_dir, f"{filename}.pkl")
        joblib.dump(self.pipeline, model_path)
        print(f"[INFO] Pipeline sauvegardé dans {model_path}")


def main():
    """
    Fonction principale pour exécuter l'entraînement et l'évaluation du modèle.
    """
    predictor_columns = [
        "diagonal",
        "height_left",
        "height_right",
        "length",
        "margin_up",
    ]
    trainer = ModelTrainer(
        predictor_columns=predictor_columns, imputer_target_column="margin_low"
    )
    trainer.load_and_split_data()
    trainer.create_pipeline()
    trainer.train_model()
    trainer.evaluate_model()
    trainer.save_model()


if __name__ == "__main__":
    main()
