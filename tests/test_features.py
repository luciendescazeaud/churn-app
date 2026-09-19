"""Tests du feature engineering.

Les deux tests qui comptent le plus sont ceux d'absence de fuite : ils
vérifient une propriété du pipeline entier, pas le comportement d'une fonction
isolée. Ce sont eux qu'il faut regarder en premier si un jour les métriques
deviennent suspectes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn import (
    LeakageError,
    align_features,
    build_features,
    feature_columns,
)

from .conftest import CUTOFF

# --- Agrégats de base ------------------------------------------------------


def test_compte_les_sessions_distinctes(events):
    df = events(
        [
            {"sessionId": 1},
            {"sessionId": 1},
            {"sessionId": 2},
        ]
    )

    f = build_features(df, cutoff=CUTOFF)

    assert f["num_sessions"].iloc[0] == 2


def test_compte_les_chansons_ecoutees(events):
    """Trois `NextSong` donnent trois chansons ; les autres pages ne comptent
    pas, car leur colonne `song` est vide."""
    df = events(
        [
            {"page": "NextSong", "song": "a"},
            {"page": "NextSong", "song": "b"},
            {"page": "NextSong", "song": "c"},
            {"page": "Home", "song": None},
            {"page": "Thumbs Up", "song": None},
        ]
    )

    f = build_features(df, cutoff=CUTOFF)

    assert f["songs_played"].iloc[0] == 3
    assert f["total_page_views"].iloc[0] == 5


def test_somme_les_durees_d_ecoute(events):
    df = events(
        [
            {"page": "NextSong", "length": 100.0},
            {"page": "NextSong", "length": 250.5},
        ]
    )

    f = build_features(df, cutoff=CUTOFF)

    assert f["total_listening_time"].iloc[0] == pytest.approx(350.5)


def test_separe_les_utilisateurs(events):
    df = events(
        [
            {"userId": "1", "page": "NextSong"},
            {"userId": "1", "page": "NextSong"},
            {"userId": "2", "page": "NextSong"},
        ]
    )

    f = build_features(df, cutoff=CUTOFF).set_index("userId")

    assert f.loc["1", "songs_played"] == 2
    assert f.loc["2", "songs_played"] == 1


# --- Absence de fuite ------------------------------------------------------


def test_les_evenements_posterieurs_au_cutoff_ne_changent_rien(events):
    """Le test le plus important de la suite.

    On construit les features sur un historique, puis on rejoue le calcul
    après avoir ajouté des événements postérieurs au cutoff. Le résultat doit
    être rigoureusement identique : si une seule feature bouge, c'est qu'une
    information du futur s'est glissée dans le calcul.
    """
    passe = [
        {"time": CUTOFF - pd.Timedelta(days=3), "page": "NextSong"},
        {"time": CUTOFF - pd.Timedelta(days=1), "page": "Thumbs Up"},
    ]
    futur = [
        {"time": CUTOFF + pd.Timedelta(days=1), "page": "NextSong"},
        {"time": CUTOFF + pd.Timedelta(days=2), "page": "Cancellation Confirmation"},
        {"time": CUTOFF + pd.Timedelta(days=5), "page": "NextSong"},
    ]

    avant = build_features(events(passe), cutoff=CUTOFF)
    apres = build_features(events(passe + futur), cutoff=CUTOFF)

    pd.testing.assert_frame_equal(avant, apres)


def test_aucune_colonne_ne_porte_un_nom_interdit(events):
    """Même en présence de pages de résiliation avant le cutoff, aucune
    feature ne doit en porter la trace dans son nom."""
    df = events(
        [
            {"page": "NextSong"},
            {"page": "Downgrade"},
            {"page": "Cancel"},
            {"page": "Submit Downgrade"},
        ]
    )

    f = build_features(df, cutoff=CUTOFF)

    interdits = [c for c in f.columns if "cancel" in c.lower() or "downgrade" in c.lower()]
    assert interdits == []


def test_le_garde_fou_peut_etre_declenche(events, monkeypatch):
    """Si un jour un bloc de features produit une colonne interdite, la
    construction doit échouer plutôt que d'entraîner un modèle biaisé."""
    import churn.features as features_module

    df = events([{"page": "NextSong"}, {"page": "Downgrade"}])

    # On neutralise la liste des pages exclues : `page_Downgrade` survit alors
    # jusqu'au contrôle final, qui doit lever.
    with pytest.raises(LeakageError):
        features_module.build_features(df, cutoff=CUTOFF, leaky_pages=())


def test_les_pages_de_resiliation_sont_exclues_des_comptages(events):
    df = events(
        [
            {"page": "NextSong"},
            {"page": "Downgrade"},
        ]
    )

    f = build_features(df, cutoff=CUTOFF)

    assert "page_NextSong" in f.columns
    assert "page_Downgrade" not in f.columns


# --- Population ------------------------------------------------------------


def test_build_features_sans_population_explicite(events):
    """Régression : `users=None` levait une erreur dans la version notebook,
    à cause d'une ligne mal indentée."""
    df = events([{"userId": "1"}, {"userId": "2"}])

    f = build_features(df, cutoff=CUTOFF, users=None)

    assert len(f) == 2


def test_build_features_restreint_a_la_population_demandee(events):
    df = events([{"userId": "1"}, {"userId": "2"}, {"userId": "3"}])

    f = build_features(df, cutoff=CUTOFF, users=["1", "2"])

    assert set(f["userId"]) == {"1", "2"}


def test_build_features_echoue_si_rien_n_est_observable(events):
    df = events([{"time": CUTOFF + pd.Timedelta(days=1)}])

    with pytest.raises(ValueError, match="Aucun événement"):
        build_features(df, cutoff=CUTOFF)


# --- Sortie ----------------------------------------------------------------


def test_aucune_valeur_manquante_en_sortie(synthetic_events):
    """Le modèle reçoit un tableau numérique dense : aucun NaN ne doit
    subsister, quelle que soit la variété des historiques."""
    from churn import filter_valid_users

    f = build_features(filter_valid_users(synthetic_events), cutoff=CUTOFF)

    assert f.isna().sum().sum() == 0


def test_aucune_colonne_temporelle_en_sortie(synthetic_events):
    from churn import filter_valid_users

    f = build_features(filter_valid_users(synthetic_events), cutoff=CUTOFF)

    temporelles = [
        c for c in f.columns if pd.api.types.is_datetime64_any_dtype(f[c])
    ]
    assert temporelles == []


def test_une_ligne_par_utilisateur(synthetic_events):
    from churn import filter_valid_users

    propres = filter_valid_users(synthetic_events)
    f = build_features(propres, cutoff=CUTOFF)

    assert f["userId"].is_unique


# --- Alignement des colonnes -----------------------------------------------


def test_align_features_cree_les_colonnes_manquantes():
    """Le cas réel : un fichier uploadé ne contient pas toutes les pages vues
    à l'entraînement, donc certaines colonnes `page_*` n'existent pas."""
    f = pd.DataFrame({"userId": ["1"], "num_sessions": [3]})

    out = align_features(f, ["num_sessions", "page_Error"])

    assert out["page_Error"].iloc[0] == 0


def test_align_features_ecarte_les_colonnes_en_trop():
    f = pd.DataFrame({"userId": ["1"], "num_sessions": [3], "inconnue": [9]})

    out = align_features(f, ["num_sessions"])

    assert "inconnue" not in out.columns


def test_align_features_respecte_l_ordre_de_reference():
    """L'ordre compte : le modèle reçoit un tableau positionnel, une
    permutation silencieuse fausserait toutes les prédictions."""
    f = pd.DataFrame({"userId": ["1"], "b": [1], "a": [2]})

    out = align_features(f, ["a", "b"])

    assert list(out.columns) == ["userId", "a", "b"]


def test_align_features_conserve_la_cible_si_elle_est_la():
    f = pd.DataFrame({"userId": ["1"], "a": [1], "target": [0]})

    out = align_features(f, ["a"])

    assert "target" in out.columns


def test_feature_columns_exclut_identifiant_et_cible():
    f = pd.DataFrame({"userId": ["1"], "b": [1], "a": [2], "target": [0]})

    assert feature_columns(f) == ["a", "b"]
