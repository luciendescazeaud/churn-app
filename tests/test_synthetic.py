"""Tests du générateur de données synthétiques.

Le générateur alimente à la fois les fixtures de tests et la démonstration de
l'application. S'il dérive, les deux dérivent avec lui sans que rien ne le
signale — d'où ces tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn import (
    SplitConfig,
    build_features,
    build_target,
    churned_users,
    eligible_users,
    filter_valid_users,
    make_events,
    validate_schema,
)
from churn.config import CHURN_PAGE, RAW_COLUMNS
from churn.synthetic import write_demo_csv


def test_le_journal_respecte_le_schema_brut():
    df = make_events(n_users=10, seed=0)

    assert list(df.columns) == list(RAW_COLUMNS)
    validate_schema(df)


def test_meme_graine_meme_resultat():
    """Sans cette garantie, un test qui passe aujourd'hui peut échouer demain
    sans qu'aucune ligne de code ait changé."""
    a = make_events(n_users=10, seed=7)
    b = make_events(n_users=10, seed=7)

    pd.testing.assert_frame_equal(a, b)


def test_graines_differentes_resultats_differents():
    a = make_events(n_users=10, seed=7)
    b = make_events(n_users=10, seed=8)

    assert not a.equals(b)


def test_le_journal_est_trie_chronologiquement():
    df = make_events(n_users=10, seed=0)

    assert df["time"].is_monotonic_increasing


def test_contient_des_sessions_deconnectees():
    """Le défaut doit être présent, sinon le test de `filter_valid_users` sur
    données réalistes ne vérifie rien."""
    df = make_events(n_users=10, seed=0, logged_out_events=25)

    assert (df["userId"] == "").sum() == 25


def test_le_nombre_d_utilisateurs_identifies_est_respecte():
    df = make_events(n_users=30, seed=0)

    assert filter_valid_users(df)["userId"].nunique() == 30


def test_les_resiliations_tombent_dans_la_fenetre_de_prediction():
    """C'est ce qui rend le jeu utilisable : la cible est observable là où le
    protocole la cherche."""
    cfg = SplitConfig()
    df = make_events(n_users=40, seed=3, config=cfg, churn_rate=0.25)

    resiliations = df.loc[df["page"] == CHURN_PAGE, "time"]

    assert len(resiliations) > 0
    assert (resiliations > cfg.train_cutoff).all()
    assert (resiliations <= cfg.final_cutoff).all()


def test_le_taux_de_churn_demande_est_respecte():
    cfg = SplitConfig()
    df = filter_valid_users(make_events(n_users=100, seed=5, config=cfg, churn_rate=0.2))

    churners = churned_users(df, cfg.train_cutoff, cfg.churn_window_days)

    assert len(churners) == pytest.approx(20, abs=3)


def test_un_resiliant_n_a_plus_d_activite_apres_sa_resiliation():
    """Propriété structurante : c'est l'arrêt d'activité que le modèle doit
    apprendre à anticiper."""
    cfg = SplitConfig()
    df = filter_valid_users(make_events(n_users=40, seed=11, config=cfg))

    resiliations = df.loc[df["page"] == CHURN_PAGE].set_index("userId")["time"]

    for user_id, cancel_time in resiliations.items():
        apres = df.loc[(df["userId"] == user_id) & (df["time"] > cancel_time)]
        assert apres.empty, f"activité après résiliation pour {user_id}"


def test_refuse_des_parametres_absurdes():
    with pytest.raises(ValueError):
        make_events(n_users=0)

    with pytest.raises(ValueError):
        make_events(churn_rate=1.5)


def test_write_demo_csv_produit_un_fichier_relisible(tmp_path):
    from churn import load_events

    path = write_demo_csv(tmp_path / "sous-dossier" / "demo.csv", n_users=5, seed=0)

    assert path.exists()
    validate_schema(load_events(path))


# --- Intégration -----------------------------------------------------------


def test_le_pipeline_complet_tourne_sur_des_donnees_synthetiques():
    """Chaîne de bout en bout : génération, nettoyage, population, cible,
    features. C'est le parcours que fera l'application."""
    cfg = SplitConfig()
    df = filter_valid_users(make_events(n_users=50, seed=42, config=cfg))

    population = eligible_users(df, cfg.train_cutoff)
    churners = churned_users(df, cfg.train_cutoff, cfg.churn_window_days)

    f = build_target(
        build_features(df, cutoff=cfg.train_cutoff, users=population), churners
    )

    assert len(f) == len(population)
    assert f["target"].isin([0, 1]).all()
    assert f["target"].sum() > 0
    assert f.isna().sum().sum() == 0
