"""Tests de l'entraînement, des métriques et de la prédiction.

L'entraînement complet est lancé **une seule fois** pour tout le module, sur
un jeu volontairement petit. L'objectif n'est pas de mesurer la performance du
modèle — elle dépend des données — mais de vérifier que la chaîne produit des
artefacts cohérents et que la prédiction les réutilise correctement.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from churn import SplitConfig, make_events
from churn.evaluation import best_threshold, classification_metrics, scan_thresholds
from churn.predict import ArtifactsError, load_artifacts, predict_from_events
from churn.train import save_artifacts, train_from_events

# --- Seuil et métriques ----------------------------------------------------


def test_scan_thresholds_couvre_les_colonnes_attendues():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.7, 0.9])

    table = scan_thresholds(y, proba, n_thresholds=20)

    assert {"threshold", "balanced_accuracy", "recall_churn", "recall_non_churn"} <= set(
        table.columns
    )
    assert table["balanced_accuracy"].between(0, 1).all()


def test_best_threshold_trouve_la_separation_parfaite():
    """Sur des probabilités parfaitement séparées, la balanced accuracy doit
    atteindre 1 et le seuil tomber entre les deux groupes."""
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.01, 0.02, 0.03, 0.90, 0.95, 0.99])

    seuil, ba = best_threshold(y, proba)

    assert ba == pytest.approx(1.0)
    assert 0.03 < seuil <= 0.90


def test_best_threshold_prefere_le_seuil_le_plus_eleve_a_egalite():
    """À performance égale, un seuil plus haut prédit moins de positifs, donc
    déclenche moins d'actions commerciales inutiles."""
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.1, 0.9, 0.9])

    seuil, ba = best_threshold(y, proba)

    assert ba == pytest.approx(1.0)
    assert seuil == pytest.approx(0.9)


def test_classification_metrics_matrice_coherente():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.8, 0.2, 0.9])

    m = classification_metrics(y, proba, threshold=0.5)
    matrice = m["confusion_matrix"]

    assert sum(matrice.values()) == len(y)
    assert matrice["true_positive"] == 1
    assert matrice["false_positive"] == 1
    assert m["n_samples"] == 4


def test_classification_metrics_rappels_dans_les_bornes():
    y = np.array([0, 1, 0, 1, 1])
    proba = np.array([0.2, 0.7, 0.4, 0.8, 0.1])

    m = classification_metrics(y, proba, threshold=0.5)

    assert 0 <= m["recall_churn"] <= 1
    assert 0 <= m["recall_non_churn"] <= 1
    assert m["balanced_accuracy"] == pytest.approx(
        (m["recall_churn"] + m["recall_non_churn"]) / 2
    )


# --- Entraînement ----------------------------------------------------------


@pytest.fixture(scope="module")
def resultat():
    """Un entraînement complet, partagé par les tests du module."""
    evenements = make_events(n_users=250, seed=7, churn_rate=0.15)
    return train_from_events(evenements, n_splits=3, seed=7)


def test_l_entrainement_produit_un_modele_utilisable(resultat):
    assert hasattr(resultat.model, "predict_proba")
    assert len(resultat.columns) > 50


def test_le_seuil_est_dans_les_bornes(resultat):
    assert 0.0 <= resultat.threshold <= 1.0


def test_les_metriques_battent_le_hasard(resultat):
    """Le générateur produit des résiliants aux comportements distincts : un
    modèle qui n'apprendrait rien signalerait une chaîne cassée."""
    assert resultat.metrics["roc_auc"] > 0.65
    assert resultat.metrics["balanced_accuracy"] > 0.6


def test_l_importance_couvre_toutes_les_features(resultat):
    assert set(resultat.importance["feature"]) == set(resultat.columns)


def test_la_provenance_est_renseignee(resultat):
    p = resultat.provenance

    assert p["n_users"] > 0
    assert p["n_churners"] > 0
    assert p["seed"] == 7
    assert p["train_cutoff"] < p["final_cutoff"]
    assert len(p["dataset_fingerprint"]) == 16
    assert "lightgbm" in p["versions"]


def test_l_entrainement_est_reproductible():
    """Même graine, mêmes données, même seuil — sans quoi les artefacts
    livrés ne seraient pas reproductibles."""
    evenements = make_events(n_users=200, seed=3, churn_rate=0.15)

    a = train_from_events(evenements, n_splits=3, seed=3)
    b = train_from_events(evenements, n_splits=3, seed=3)

    assert a.threshold == pytest.approx(b.threshold)
    assert a.metrics["roc_auc"] == pytest.approx(b.metrics["roc_auc"])


def test_l_entrainement_refuse_un_jeu_sans_resiliations():
    evenements = make_events(n_users=40, seed=0, churn_rate=0.0)

    with pytest.raises(ValueError, match="Trop peu de résiliations"):
        train_from_events(evenements, n_splits=5)


# --- Artefacts -------------------------------------------------------------


@pytest.fixture(scope="module")
def dossier_artefacts(resultat, tmp_path_factory):
    chemin = tmp_path_factory.mktemp("artifacts")
    save_artifacts(resultat, chemin)
    return chemin


def test_save_artifacts_ecrit_tous_les_fichiers(dossier_artefacts):
    attendus = {
        "model.joblib",
        "feature_columns.json",
        "metrics.json",
        "feature_importance.csv",
        "threshold_scan.csv",
    }

    assert attendus <= {f.name for f in dossier_artefacts.iterdir()}


def test_metrics_json_est_lisible_et_complet(dossier_artefacts):
    contenu = json.loads((dossier_artefacts / "metrics.json").read_text(encoding="utf-8"))

    assert "threshold" in contenu
    assert "metrics" in contenu
    assert "provenance" in contenu


def test_load_artifacts_retrouve_le_seuil(resultat, dossier_artefacts):
    charges = load_artifacts(dossier_artefacts)

    assert charges.threshold == pytest.approx(resultat.threshold)
    assert charges.columns == resultat.columns


def test_load_artifacts_nomme_les_fichiers_manquants(tmp_path):
    with pytest.raises(ArtifactsError) as err:
        load_artifacts(tmp_path)

    assert "model.joblib" in str(err.value)


# --- Prédiction ------------------------------------------------------------


@pytest.fixture(scope="module")
def artefacts(dossier_artefacts):
    return load_artifacts(dossier_artefacts)


def test_predire_sur_un_journal_jamais_vu(artefacts):
    cfg = SplitConfig()
    nouveau = make_events(n_users=30, seed=999, config=cfg)

    p = predict_from_events(nouveau, artefacts, cutoff=cfg.train_cutoff)

    assert len(p) == 30
    assert p["probability"].between(0, 1).all()
    assert set(p["prediction"].unique()) <= {0, 1}


def test_la_prediction_applique_le_seuil_des_artefacts(artefacts):
    cfg = SplitConfig()
    nouveau = make_events(n_users=30, seed=1001, config=cfg)

    p = predict_from_events(nouveau, artefacts, cutoff=cfg.train_cutoff)

    attendu = (p["probability"] >= artefacts.threshold).astype(int)
    pd.testing.assert_series_equal(p["prediction"], attendu, check_names=False)


def test_la_prediction_est_triee_par_probabilite(artefacts):
    cfg = SplitConfig()
    nouveau = make_events(n_users=30, seed=1002, config=cfg)

    p = predict_from_events(nouveau, artefacts, cutoff=cfg.train_cutoff)

    assert p["probability"].is_monotonic_decreasing


def test_la_prediction_survit_a_des_pages_absentes(artefacts, events):
    """Le cas réel d'un fichier uploadé : il ne contient qu'une poignée de
    pages, donc presque toutes les colonnes `page_*` manquent. C'est
    `align_features` qui doit rattraper le coup."""
    from .conftest import CUTOFF

    maigre = events(
        [
            {"userId": "1", "page": "NextSong", "time": CUTOFF - pd.Timedelta(days=2)},
            {"userId": "1", "page": "NextSong", "time": CUTOFF - pd.Timedelta(days=1)},
        ]
    )

    p = predict_from_events(maigre, artefacts, cutoff=CUTOFF)

    assert len(p) == 1
    assert 0 <= p["probability"].iloc[0] <= 1


def test_la_prediction_refuse_un_journal_sans_utilisateur(artefacts, events):
    vide = events([{"userId": ""}, {"userId": None}])

    with pytest.raises(ValueError, match="Aucun utilisateur identifié"):
        predict_from_events(vide, artefacts)


def test_la_prediction_refuse_un_schema_incomplet(artefacts, events):
    from churn import SchemaError

    incomplet = events([{"userId": "1"}]).drop(columns=["sessionId"])

    with pytest.raises(SchemaError):
        predict_from_events(incomplet, artefacts)
