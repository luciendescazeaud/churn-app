"""Tests des fonctions d'import et de filtrage.

C'est le cœur de ce que l'énoncé demande de couvrir. Chaque test vérifie une
règle métier précise, jamais la simple présence d'une colonne.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn import (
    LeakageError,
    SchemaError,
    SplitConfig,
    assert_no_leaky_columns,
    build_target,
    churned_users,
    drop_leaky_pages,
    eligible_users,
    filter_valid_users,
    find_leaky_columns,
    load_events,
    restrict_to_users,
    truncate_at,
    validate_schema,
)

from .conftest import CUTOFF

# --- Import ----------------------------------------------------------------


def test_load_events_lit_un_csv(tmp_path, events):
    """Un CSV relu doit retrouver ses types temporels, perdus par le format."""
    path = tmp_path / "events.csv"
    events([{"userId": "1"}, {"userId": "2"}]).to_csv(path, index=False)

    df = load_events(path)

    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["time"])
    assert pd.api.types.is_datetime64_any_dtype(df["registration"])


def test_load_events_lit_un_parquet(tmp_path, events):
    path = tmp_path / "events.parquet"
    events([{"userId": "1"}]).to_parquet(path)

    assert len(load_events(path)) == 1


def test_load_events_uniformise_le_type_de_userid(tmp_path, events):
    """Un userId entier dans le fichier doit ressortir en chaîne.

    Les deux formes coexistent dans les données réelles, et une comparaison
    entre entier et chaîne échoue silencieusement.
    """
    path = tmp_path / "events.parquet"
    df = events([{"userId": "1"}])
    df["userId"] = [42]
    df.to_parquet(path)

    assert load_events(path)["userId"].iloc[0] == "42"


def test_load_events_signale_un_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_events(tmp_path / "absent.csv")


def test_load_events_refuse_une_extension_inconnue(tmp_path):
    path = tmp_path / "events.json"
    path.write_text("{}")

    with pytest.raises(SchemaError):
        load_events(path)


# --- Validation de schéma --------------------------------------------------


def test_validate_schema_accepte_un_journal_complet(events):
    df = events([{"userId": "1"}])

    assert validate_schema(df) is df


def test_validate_schema_nomme_les_colonnes_manquantes(events):
    df = events([{"userId": "1"}]).drop(columns=["sessionId", "location"])

    with pytest.raises(SchemaError) as err:
        validate_schema(df)

    message = str(err.value)
    assert "sessionId" in message
    assert "location" in message


def test_validate_schema_refuse_une_colonne_temporelle_non_convertie(events):
    """Un `time` resté en chaîne casserait toutes les comparaisons de dates."""
    df = events([{"userId": "1"}])
    df["time"] = df["time"].astype(str)

    with pytest.raises(SchemaError, match="time"):
        validate_schema(df)


# --- Filtrage des utilisateurs ---------------------------------------------


def test_filter_valid_users_retire_les_chaines_vides(events):
    """Le défaut central des données : les sessions déconnectées portent une
    chaîne vide, pas une valeur manquante. Un filtre sur `isna()` seul les
    laisserait passer et créerait un utilisateur fantôme."""
    df = events([{"userId": "1"}, {"userId": ""}, {"userId": "2"}])

    out = filter_valid_users(df)

    assert set(out["userId"]) == {"1", "2"}


def test_filter_valid_users_retire_aussi_les_valeurs_manquantes(events):
    df = events([{"userId": "1"}, {"userId": None}])

    assert set(filter_valid_users(df)["userId"]) == {"1"}


def test_filter_valid_users_retire_les_chaines_d_espaces(events):
    df = events([{"userId": "1"}, {"userId": "   "}])

    assert set(filter_valid_users(df)["userId"]) == {"1"}


def test_filter_valid_users_ne_modifie_pas_l_entree(events):
    df = events([{"userId": "1"}, {"userId": ""}])
    avant = len(df)

    filter_valid_users(df)

    assert len(df) == avant


def test_restrict_to_users_ne_garde_que_la_population_demandee(events):
    df = events([{"userId": "1"}, {"userId": "2"}, {"userId": "3"}])

    out = restrict_to_users(df, ["1", "3"])

    assert set(out["userId"]) == {"1", "3"}


def test_restrict_to_users_compare_sur_des_chaines(events):
    """La population peut arriver en entiers depuis un `unique()` sur parquet."""
    df = events([{"userId": "1"}, {"userId": "2"}])

    assert set(restrict_to_users(df, [1])["userId"]) == {"1"}


# --- Troncature temporelle -------------------------------------------------


def test_truncate_at_inclut_la_borne(events):
    """Un événement tombant exactement au cutoff appartient au passé
    observable : il doit être conservé."""
    df = events(
        [
            {"time": CUTOFF - pd.Timedelta(days=1)},
            {"time": CUTOFF},
            {"time": CUTOFF + pd.Timedelta(seconds=1)},
        ]
    )

    out = truncate_at(df, CUTOFF)

    assert len(out) == 2
    assert out["time"].max() == CUTOFF


def test_truncate_at_peut_tout_ecarter(events):
    df = events([{"time": CUTOFF + pd.Timedelta(days=1)}])

    assert truncate_at(df, CUTOFF).empty


# --- Pages de résiliation --------------------------------------------------


def test_drop_leaky_pages_retire_les_pages_de_resiliation(events):
    df = events(
        [
            {"page": "NextSong"},
            {"page": "Cancel"},
            {"page": "Cancellation Confirmation"},
            {"page": "Downgrade"},
            {"page": "Thumbs Up"},
        ]
    )

    out = drop_leaky_pages(df)

    assert set(out["page"]) == {"NextSong", "Thumbs Up"}


# --- Population éligible ---------------------------------------------------


def test_eligible_users_exclut_ceux_sans_historique_observable(events):
    """Un utilisateur dont toute l'activité est postérieure au cutoff n'a rien
    d'observable : on ne peut rien prédire pour lui."""
    df = events(
        [
            {"userId": "1", "time": CUTOFF - pd.Timedelta(days=2)},
            {"userId": "2", "time": CUTOFF + pd.Timedelta(days=2)},
        ]
    )

    assert set(eligible_users(df, CUTOFF)) == {"1"}


# --- Construction de la cible ----------------------------------------------


def test_churned_users_detecte_une_resiliation_dans_la_fenetre(events):
    df = events(
        [
            {
                "userId": "1",
                "page": "Cancellation Confirmation",
                "time": CUTOFF + pd.Timedelta(days=3),
            }
        ]
    )

    assert set(churned_users(df, CUTOFF, window_days=10)) == {"1"}


def test_churned_users_ignore_une_resiliation_posterieure_a_la_fenetre(events):
    """Au-delà de l'horizon, la résiliation n'est plus ce que le modèle
    cherche à prédire."""
    df = events(
        [
            {
                "userId": "1",
                "page": "Cancellation Confirmation",
                "time": CUTOFF + pd.Timedelta(days=11),
            }
        ]
    )

    assert len(churned_users(df, CUTOFF, window_days=10)) == 0


def test_churned_users_ignore_une_resiliation_anterieure_au_cutoff(events):
    """Une résiliation déjà survenue appartient au passé observable, pas au
    futur à prédire."""
    df = events(
        [
            {
                "userId": "1",
                "page": "Cancellation Confirmation",
                "time": CUTOFF - pd.Timedelta(days=1),
            }
        ]
    )

    assert len(churned_users(df, CUTOFF, window_days=10)) == 0


def test_churned_users_borne_inferieure_stricte_borne_superieure_incluse(events):
    df = events(
        [
            {"userId": "1", "page": "Cancellation Confirmation", "time": CUTOFF},
            {
                "userId": "2",
                "page": "Cancellation Confirmation",
                "time": CUTOFF + pd.Timedelta(days=10),
            },
        ]
    )

    assert set(churned_users(df, CUTOFF, window_days=10)) == {"2"}


def test_churned_users_ignore_les_autres_pages(events):
    df = events(
        [{"userId": "1", "page": "Downgrade", "time": CUTOFF + pd.Timedelta(days=1)}]
    )

    assert len(churned_users(df, CUTOFF, window_days=10)) == 0


def test_build_target_marque_les_resiliants(events):
    features = pd.DataFrame({"userId": ["1", "2", "3"]})

    out = build_target(features, ["1", "3"])

    assert out["target"].tolist() == [1, 0, 1]


def test_build_target_compare_sur_des_chaines():
    """Le jeu de features peut porter des userId entiers, la liste de
    résiliants des chaînes, ou l'inverse."""
    features = pd.DataFrame({"userId": [1, 2]})

    assert build_target(features, ["1"])["target"].tolist() == [1, 0]


# --- Garde-fou anti-fuite --------------------------------------------------


def test_find_leaky_columns_repere_les_jetons_interdits():
    colonnes = ["num_sessions", "page_Downgrade", "page_Cancel", "songs_played"]

    assert set(find_leaky_columns(colonnes)) == {"page_Downgrade", "page_Cancel"}


def test_find_leaky_columns_ignore_la_casse():
    assert find_leaky_columns(["page_CANCEL"]) == ["page_CANCEL"]


def test_assert_no_leaky_columns_passe_sur_un_jeu_propre():
    assert assert_no_leaky_columns(["num_sessions", "songs_played"]) is None


def test_assert_no_leaky_columns_echoue_et_nomme_la_colonne():
    with pytest.raises(LeakageError, match="page_Downgrade"):
        assert_no_leaky_columns(["num_sessions", "page_Downgrade"])


# --- Protocole temporel ----------------------------------------------------


def test_split_config_recule_le_cutoff_d_une_fenetre():
    cfg = SplitConfig(final_cutoff="2018-11-20", churn_window_days=10)

    assert cfg.train_cutoff == pd.Timestamp("2018-11-10")


def test_split_config_fenetre_d_entrainement_finit_au_cutoff_de_test():
    """Invariant central du protocole : entraînement et inférence observent le
    même horizon, et aucun événement postérieur au cutoff d'entraînement
    n'alimente les features d'entraînement."""
    cfg = SplitConfig(final_cutoff="2018-11-20", churn_window_days=10)

    assert cfg.train_churn_end == cfg.final_cutoff


def test_split_config_refuse_une_fenetre_nulle_ou_negative():
    with pytest.raises(ValueError):
        SplitConfig(churn_window_days=0)


def test_split_config_accepte_une_date_en_chaine():
    assert SplitConfig(final_cutoff="2018-11-20").final_cutoff == pd.Timestamp(
        "2018-11-20"
    )
