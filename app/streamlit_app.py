"""Application Streamlit de prédiction du churn.

Elle ne fait que deux choses : présenter les performances du modèle, et
appliquer ce modèle à un journal d'événements fourni par l'utilisateur.

Toute la logique vit dans le package `churn`. Cette couche n'est qu'une
interface : elle charge des artefacts, appelle `predict_from_events`, et
affiche le résultat. C'est délibéré — le code métier est testé, l'interface
ne l'est pas, donc l'interface doit contenir le moins de logique possible.

Lancement :

    make app
    # ou
    PYTHONPATH=src streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permet de lancer l'application sans que le package soit installé, en
# complément de l'installation classique dans l'image Docker.
_RACINE = Path(__file__).resolve().parent.parent
if str(_RACINE / "src") not in sys.path:
    sys.path.insert(0, str(_RACINE / "src"))

from churn import SchemaError, SplitConfig, make_events  # noqa: E402
from churn.predict import (  # noqa: E402
    Artifacts,
    ArtifactsError,
    load_artifacts,
    predict_from_events,
)

ARTEFACTS_PAR_DEFAUT = _RACINE / "artifacts" / "demo"

st.set_page_config(
    page_title="Prédiction du churn",
    page_icon="📉",
    layout="wide",
)


# --- Chargement ------------------------------------------------------------


@st.cache_resource(show_spinner="Chargement du modèle…")
def charger_artefacts(dossier: str) -> Artifacts:
    """Charger le modèle une fois pour toutes.

    `cache_resource` conserve l'objet entre les interactions : sans lui, le
    modèle serait relu à chaque clic, Streamlit réexécutant le script entier
    à la moindre action.
    """
    return load_artifacts(dossier)


@st.cache_data(show_spinner="Génération du jeu d'exemple…")
def journal_de_demonstration(n_utilisateurs: int, graine: int) -> pd.DataFrame:
    """Journal synthétique, produit par la fonction que testent les tests.

    Aucune donnée réelle n'est embarquée dans l'application : le jeu
    d'exemple est tiré au sort à la volée, de façon reproductible.
    """
    return make_events(n_users=n_utilisateurs, seed=graine)


# --- Pages -----------------------------------------------------------------


def page_performance(artefacts: Artifacts) -> None:
    st.header("Performance du modèle")
    st.caption(
        "Mesures obtenues par validation croisée : chaque utilisateur est "
        "prédit par un modèle qui ne l'a pas vu à l'entraînement."
    )

    m = artefacts.metrics["metrics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Balanced accuracy", f"{m['balanced_accuracy']:.3f}")
    c2.metric("AUC", f"{m['roc_auc']:.3f}")
    c3.metric("Rappel résiliants", f"{m['recall_churn']:.1%}")
    c4.metric("Rappel fidèles", f"{m['recall_non_churn']:.1%}")

    st.divider()

    gauche, droite = st.columns([2, 3])

    with gauche:
        st.subheader("Matrice de confusion")
        mc = m["confusion_matrix"]
        st.dataframe(
            pd.DataFrame(
                [
                    [mc["true_negative"], mc["false_positive"]],
                    [mc["false_negative"], mc["true_positive"]],
                ],
                index=["Fidèle (réel)", "Résiliant (réel)"],
                columns=["Fidèle (prédit)", "Résiliant (prédit)"],
            ),
        )

        st.caption(
            f"Seuil de décision : **{m['threshold']:.4f}**. "
            f"Le modèle signale {m['predicted_positive_rate']:.1%} des "
            f"utilisateurs, pour {m['actual_positive_rate']:.1%} de "
            "résiliations réelles."
        )

        with st.expander("Pourquoi signaler plus que le taux réel ?"):
            st.markdown(
                "La balanced accuracy traite les deux classes à parts égales. "
                "Pour rattraper une classe rare, le modèle accepte de "
                "surprédire : il vaut mieux contacter quelques fidèles à tort "
                "que de laisser partir des résiliants. Un objectif commercial "
                "différent — un budget de rétention limité, par exemple — "
                "appellerait un autre seuil, et la courbe ci-contre permet "
                "de le choisir."
            )

    with droite:
        st.subheader("Choix du seuil")
        courbe = charger_courbe_seuil()
        if courbe is not None:
            st.line_chart(
                courbe.set_index("threshold")["balanced_accuracy"],
                height=300,
                x_label="Seuil de décision",
                y_label="Balanced accuracy",
            )
            st.caption(
                "Balanced accuracy en fonction du seuil. Le maximum donne le "
                f"seuil retenu, {m['threshold']:.4f}."
            )
        else:
            st.info("Courbe indisponible : `threshold_scan.csv` est absent.")

    st.divider()

    st.subheader("Features les plus utilisées")
    if artefacts.importance.empty:
        st.info("Aucune importance de features dans les artefacts.")
    else:
        top = artefacts.importance.head(15).sort_values("importance")
        st.bar_chart(
            top.set_index("feature")["importance"],
            horizontal=True,
            height=420,
            x_label="Nombre d'utilisations dans les arbres",
            y_label="",
        )
        st.caption(
            "Nombre de fois où chaque feature sert à découper un arbre. "
            "Une feature très utilisée n'est pas nécessairement causale."
        )

    with st.expander("Provenance de ce modèle"):
        st.json(artefacts.metrics["provenance"])


def page_prediction(artefacts: Artifacts) -> None:
    st.header("Prédire sur un journal d'événements")
    st.caption(
        "Le fichier suit exactement le chemin de l'entraînement : validation "
        "du schéma, nettoyage, feature engineering, alignement des colonnes, "
        "puis modèle."
    )

    source = st.radio(
        "Source des données",
        ["Jeu d'exemple", "Importer un fichier"],
        horizontal=True,
        help=(
            "Le jeu d'exemple est synthétique et généré à la volée. Aucune "
            "donnée de compétition n'est embarquée dans l'application."
        ),
    )

    journal: pd.DataFrame | None = None

    if source == "Jeu d'exemple":
        c1, c2 = st.columns(2)
        n_utilisateurs = c1.slider("Nombre d'utilisateurs", 20, 500, 100, step=20)
        graine = c2.number_input("Graine du tirage", 0, 9999, 0, step=1)
        journal = journal_de_demonstration(n_utilisateurs, int(graine))

    else:
        fichier = st.file_uploader(
            "Journal d'événements (.csv ou .parquet)",
            type=["csv", "parquet"],
            help="Le fichier doit porter les colonnes des logs bruts.",
        )
        if fichier is not None:
            try:
                journal = (
                    pd.read_csv(fichier)
                    if fichier.name.endswith(".csv")
                    else pd.read_parquet(fichier)
                )
                journal = _normaliser(journal)
            except Exception as erreur:  # noqa: BLE001 — message à l'utilisateur
                st.error(f"Lecture impossible : {erreur}")
                journal = None

    if journal is None:
        st.info("Choisis un jeu d'exemple ou importe un fichier pour continuer.")
        return

    st.success(f"{len(journal):,} événements chargés.".replace(",", " "))

    with st.expander("Aperçu des données brutes"):
        st.dataframe(journal.head(20))

    cutoff = st.date_input(
        "Instant de prédiction",
        value=SplitConfig().train_cutoff.date(),
        help=(
            "Seuls les événements antérieurs à cette date alimentent les "
            "features. C'est la garantie qu'aucune information future "
            "n'entre dans le calcul."
        ),
    )

    if not st.button("Lancer la prédiction", type="primary"):
        return

    try:
        with st.spinner("Construction des features et prédiction…"):
            resultat = predict_from_events(
                journal, artefacts, cutoff=pd.Timestamp(cutoff)
            )
    except SchemaError as erreur:
        st.error(f"Schéma invalide — {erreur}")
        return
    except ValueError as erreur:
        st.error(str(erreur))
        return

    _afficher_resultat(resultat, artefacts)


def _afficher_resultat(resultat: pd.DataFrame, artefacts: Artifacts) -> None:
    signales = int(resultat["prediction"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Utilisateurs analysés", len(resultat))
    c2.metric("Signalés à risque", signales)
    c3.metric("Taux de signalement", f"{resultat['prediction'].mean():.1%}")

    gauche, droite = st.columns([3, 2])

    with gauche:
        st.subheader("Utilisateurs les plus à risque")
        affichage = resultat.head(25).copy()
        affichage["probability"] = affichage["probability"].map("{:.1%}".format)
        affichage["prediction"] = affichage["prediction"].map(
            {1: "À risque", 0: "Fidèle"}
        )
        st.dataframe(
            affichage.rename(
                columns={
                    "userId": "Utilisateur",
                    "probability": "Probabilité",
                    "prediction": "Décision",
                }
            ),
            hide_index=True,
        )

    with droite:
        st.subheader("Distribution des probabilités")
        paliers = pd.cut(resultat["probability"], bins=20)
        histogramme = (
            resultat.groupby(paliers, observed=True)
            .size()
            .rename("Utilisateurs")
            .reset_index()
        )
        histogramme["Probabilité"] = histogramme["probability"].map(
            lambda intervalle: f"{intervalle.mid:.2f}"
        )
        st.bar_chart(
            histogramme.set_index("Probabilité")["Utilisateurs"],
            height=300,
            x_label="Probabilité de résiliation",
            y_label="Utilisateurs",
        )
        st.caption(f"Seuil appliqué : {artefacts.threshold:.4f}")

    st.download_button(
        "Télécharger les prédictions (CSV)",
        resultat.to_csv(index=False).encode("utf-8"),
        file_name="predictions_churn.csv",
        mime="text/csv",
    )


def page_a_propos(artefacts: Artifacts) -> None:
    st.header("À propos")

    st.markdown(
        """
Cette application industrialise un projet de prédiction de churn à partir de
journaux d'événements d'un service de streaming musical.

#### Le protocole anti-fuite

Les features d'un utilisateur ne sont construites qu'à partir de ses
événements **antérieurs à l'instant de prédiction**, et la cible est observée
dans la fenêtre de dix jours qui suit. Trois garde-fous se superposent :

1. le journal est tronqué au cutoff avant tout calcul ;
2. les pages de résiliation et de rétrogradation sont exclues des comptages ;
3. un contrôle final rejette toute feature dont le nom trahit la résiliation.

Un test vérifie cette propriété de bout en bout : ajouter des événements
postérieurs au cutoff ne doit changer **aucune** valeur du jeu de features.

#### Le seuil de décision

Le modèle produit une probabilité ; la décision suppose un seuil. Comme la
cible est rare, le seuil par défaut de 0,5 ne signalerait presque personne.
Il est donc choisi pour maximiser la balanced accuracy, sur des prédictions
hors échantillon, et **sauvegardé avec le modèle** — c'est ce qui garantit
que les prédictions affichées ici correspondent aux métriques annoncées.

#### Les données

Aucune donnée de compétition n'est redistribuée. Le jeu d'exemple est
synthétique, généré à la volée par la même fonction que celle qui alimente
les tests.
"""
    )

    provenance = artefacts.metrics.get("provenance", {})
    if provenance.get("n_users"):
        st.info(
            f"Modèle entraîné sur {provenance['n_users']:,} utilisateurs, "
            f"dont {provenance.get('n_churners', 0):,} résiliations, "
            f"avec {provenance.get('n_features', '?')} features.".replace(",", " ")
        )


# --- Utilitaires -----------------------------------------------------------


def _normaliser(journal: pd.DataFrame) -> pd.DataFrame:
    """Reconvertir les colonnes temporelles perdues par le format CSV."""
    sortie = journal.copy()
    for colonne in ("time", "registration"):
        if colonne in sortie.columns:
            sortie[colonne] = pd.to_datetime(sortie[colonne], errors="coerce")
    if "userId" in sortie.columns:
        sortie["userId"] = sortie["userId"].astype("string").fillna("").astype(object)
    return sortie


@st.cache_data
def charger_courbe_seuil() -> pd.DataFrame | None:
    chemin = Path(st.session_state.get("dossier_artefacts", ARTEFACTS_PAR_DEFAUT))
    fichier = chemin / "threshold_scan.csv"
    return pd.read_csv(fichier) if fichier.exists() else None


# --- Point d'entrée --------------------------------------------------------


def main() -> None:
    st.sidebar.title("📉 Prédiction du churn")

    dossier = st.sidebar.text_input(
        "Dossier des artefacts",
        value=str(ARTEFACTS_PAR_DEFAUT),
        help="Dossier produit par `make train-demo` ou `make train`.",
    )
    st.session_state["dossier_artefacts"] = dossier

    try:
        artefacts = charger_artefacts(dossier)
    except ArtifactsError as erreur:
        st.error(str(erreur))
        st.code("make train-demo", language="bash")
        st.stop()

    if artefacts.trained_at:
        st.sidebar.caption(f"Modèle entraîné le {artefacts.trained_at[:10]}")
    if artefacts.git_commit:
        st.sidebar.caption(f"Commit `{artefacts.git_commit[:8]}`")

    page = st.sidebar.radio(
        "Page",
        ["Performance du modèle", "Prédire sur un journal", "À propos"],
        label_visibility="collapsed",
    )

    if page == "Performance du modèle":
        page_performance(artefacts)
    elif page == "Prédire sur un journal":
        page_prediction(artefacts)
    else:
        page_a_propos(artefacts)


if __name__ == "__main__":
    main()
