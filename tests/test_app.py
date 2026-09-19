"""Tests de l'application Streamlit.

Une application qui ne démarre pas est un échec aussi grave qu'un test rouge,
et rien d'autre ne le détecterait : le reste de la suite teste le package,
jamais l'interface.

`AppTest` exécute le script Streamlit sans navigateur et sans serveur, puis
donne accès aux composants produits et aux exceptions levées. On peut ainsi
cliquer sur un bouton et vérifier ce qui s'affiche, en quelques secondes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip(
    "streamlit", reason="Streamlit n'est installé que via l'extra `app`"
)
from streamlit.testing.v1 import AppTest  # noqa: E402

from churn import make_events  # noqa: E402
from churn.train import save_artifacts, train_from_events  # noqa: E402

APPLICATION = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def artefacts_demo(tmp_path_factory) -> str:
    """Un jeu d'artefacts minimal, entraîné une fois pour tout le module."""
    dossier = tmp_path_factory.mktemp("artefacts_app")
    resultat = train_from_events(
        make_events(n_users=150, seed=5, churn_rate=0.2), n_splits=3, seed=5
    )
    save_artifacts(resultat, dossier)
    return str(dossier)


@pytest.fixture
def application(artefacts_demo) -> AppTest:
    """L'application, pointée vers les artefacts de test et déjà démarrée."""
    app = AppTest.from_file(APPLICATION, default_timeout=180)
    app.run()
    app.sidebar.text_input[0].set_value(artefacts_demo).run()
    return app


# --- Démarrage -------------------------------------------------------------


def test_l_application_demarre_sans_exception(application):
    assert not application.exception


def test_la_page_d_accueil_affiche_les_metriques(application):
    labels = {m.label for m in application.metric}

    assert "Balanced accuracy" in labels
    assert "AUC" in labels


def test_les_trois_pages_sont_proposees(application):
    assert len(application.sidebar.radio[0].options) == 3


def test_sans_artefacts_l_application_explique_quoi_faire(tmp_path):
    """Le cas de quelqu'un qui clone le dépôt et lance l'application avant
    d'avoir entraîné quoi que ce soit. Il doit lire la commande à taper, pas
    une trace Python."""
    app = AppTest.from_file(APPLICATION, default_timeout=120)
    app.run()
    app.sidebar.text_input[0].set_value(str(tmp_path / "vide")).run()

    assert not app.exception
    assert app.error, "aucun message d'erreur affiché"
    assert "train" in app.error[0].value


# --- Navigation ------------------------------------------------------------


def test_la_page_de_prediction_s_affiche(application):
    application.sidebar.radio[0].set_value("Prédire sur un journal").run()

    assert not application.exception
    assert any("prédiction" in b.label.lower() for b in application.button)


def test_la_page_a_propos_s_affiche(application):
    application.sidebar.radio[0].set_value("À propos").run()

    assert not application.exception


# --- Prédiction de bout en bout --------------------------------------------


def test_predire_sur_le_jeu_d_exemple(application):
    """Le parcours complet que fera le correcteur : ouvrir l'application,
    aller sur la page de prédiction, cliquer, lire un résultat."""
    application.sidebar.radio[0].set_value("Prédire sur un journal").run()
    application.button[0].click().run()

    assert not application.exception

    mesures = {m.label: m.value for m in application.metric}
    assert "Utilisateurs analysés" in mesures
    assert int(mesures["Utilisateurs analysés"]) > 0
    assert application.dataframe, "aucun tableau de résultats affiché"


def test_le_nombre_d_utilisateurs_suit_le_curseur(application):
    application.sidebar.radio[0].set_value("Prédire sur un journal").run()
    application.slider[0].set_value(40).run()
    application.button[0].click().run()

    assert not application.exception

    mesures = {m.label: m.value for m in application.metric}
    assert int(mesures["Utilisateurs analysés"]) == 40
