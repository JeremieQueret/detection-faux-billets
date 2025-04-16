# streamlit_app.py
import sys
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

sys.path.append(os.path.abspath("./src"))

# Chemin direct vers le modèle à l'intérieur du conteneur Docker
model_path = "./model/pipeline_logit.pkl"

try:
    pipeline = joblib.load(model_path)
    print(f"[INFO] Modèle chargé avec succès depuis : {model_path}")

except FileNotFoundError:
    st.error(
        f"Erreur: Le fichier de modèle '{model_path}' n'a pas été trouvé. Assurez-vous que le modèle a été entraîné et sauvegardé."
    )
    st.stop()


def predict(data: pd.DataFrame):
    """Effectue les prédictions sur les données fournies."""
    try:
        predictions = pipeline.predict(data)
        return predictions
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None  # Retourner None en cas d'erreur pour gérer la suite du code


def afficher_informations():
    st.title("Informations sur le projet")

    st.header("Contexte du projet")
    st.markdown(
        """
        L’Organisation nationale de lutte contre le faux-monnayage (ONCFM) est
        une organisation publique ayant pour objectif de mettre en place des
        méthodes d’identification des contrefaçons des billets en euros. Dans le
        cadre de cette lutte, nous souhaitons mettre en place un algorithme qui
        soit capable de différencier automatiquement les vrais des faux billets.
        """
    )

    st.header("Objectifs")
    st.markdown(
        """
        Lorsqu’un billet arrive, nous avons une machine qui consigne l’ensemble
        de ses caractéristiques géométriques. Au fil de nos années de lutte, nous
        avons observé des différences de dimensions entre les vrais et les faux
        billets. Ces différences sont difficilement visibles à l’oeil nu, mais une
        machine devrait sans problème arriver à les différencier.
        Ainsi, il faudrait construire un algorithme qui, à partir des caractéristiques
        géométriques d’un billet, serait capable de définir si ce dernier est un vrai
        ou un faux billet.
        """
    )
    st.header("Modèle de données")
    st.markdown(
        """
        Nous disposons actuellement de six informations géométriques sur un
        billet :
        - length : la longueur du billet (en mm) ;
        - height_left : la hauteur du billet (mesurée sur le côté gauche, en mm) ;
        - height_right : la hauteur du billet (mesurée sur le côté droit, en mm) ;
        - margin_up : la marge entre le bord supérieur du billet et l'image de celui-ci (en mm) ;
        - margin_low : la marge entre le bord inférieur du billet et l'image de celui-ci (en mm) ;
        - diagonal : la diagonale du billet (en mm).
        Ces informations sont celles avec lesquelles l’algorithme devra opérer.
        """
    )

    st.header("Fonctionnement général")
    st.markdown(
        """
        **Cette application a une visée pédagogique et répond aux exigences du projet 12 du parcours Data Analyst.**

        L’algorithme peut traiter un fichier en entrée contenant les dimensions de plusieurs billets, et déterminer le type de chaque billet en se basant uniquement sur ces dimensions.

        Cependant, le fichier fourni par l'utilisateur n’est pas modifié. Concrètement, cela signifie que l’imputation des données manquantes — **imposée par le cahier des charges du projet** — ne porte que sur la colonne `margin_low`, et que les données imputées ne sont pas réintégrées dans le fichier d’origine.
        """
    )


def main():
    st.sidebar.title("Navigation")
    menu = ["Détection de faux billets", "Informations sur le projet"]
    choice = st.sidebar.selectbox("Choisir une page", menu)

    if choice == "Détection de faux billets":
        st.title("Détection de faux billets")
        st.subheader(
            "Téléchargez un fichier CSV contenant les caractéristiques des billets pour la détection."
        )
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep=";")
                st.subheader("Aperçu des données chargées:")
                st.dataframe(df.head())

                if st.button("Prédire"):
                    predictions = predict(df)

                    if predictions is not None:
                        df["prediction"] = predictions
                        df["is_genuine"] = df["prediction"].apply(
                            lambda x: "Vrai" if x == 1 else "Faux"
                        )

                        st.subheader("Résultats des prédictions:")
                        st.dataframe(df)

                        # Calcul des occurrences de chaque prédiction
                        prediction_counts = (
                            df["is_genuine"].value_counts().reset_index()
                        )
                        prediction_counts.columns = [
                            "Type de billet",
                            "Nombre de billets",
                        ]
                        prediction_counts["Type de billet"] = prediction_counts[
                            "Type de billet"
                        ].map({"Faux": "Faux", "Vrai": "Vrai"})

                        # Création du graphique avec Plotly Express
                        fig = px.bar(
                            prediction_counts,
                            x="Type de billet",
                            y="Nombre de billets",
                            color="Type de billet",
                            color_discrete_sequence=px.colors.qualitative.Set1,
                            title="Répartition des billets détectés",
                            labels={
                                "Nombre de billets": "Nombre de billets",
                                "Type de billet": "Type de billet",
                            },
                        )

                        # Affichage du graphique dans Streamlit
                        st.plotly_chart(fig)

            except pd.errors.EmptyDataError:
                st.error("Erreur: Le fichier CSV est vide.")
            except pd.errors.ParserError:
                st.error(
                    "Erreur: Impossible de lire le fichier CSV. Vérifiez le format et le séparateur (;)."
                )
            except Exception as e:
                st.error(
                    f"Une erreur inattendue s'est produite lors du chargement du fichier: {e}"
                )

    elif choice == "Informations sur le projet":
        afficher_informations()


if __name__ == "__main__":
    main()
