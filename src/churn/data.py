"""Import, validation et filtrage des logs d'événements.

Ce module contient les fonctions que l'énoncé demande de tester : chargement
d'un fichier de logs, contrôle de schéma, filtrage des utilisateurs, troncature
temporelle, exclusion des pages de résiliation et construction de la cible.

Toutes les fonctions sont pures : elles renvoient un nouvel objet et ne
modifient jamais leur argument.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from .config import (
    CHURN_PAGE,
    DATETIME_COLUMNS,
    FORBIDDEN_TOKENS,
    LEAKY_PAGES,
    REQUIRED_COLUMNS,
)


class SchemaError(ValueError):
    """Le fichier fourni ne respecte pas le schéma attendu des logs bruts."""


class LeakageError(AssertionError):
    """Une colonne du jeu de features trahit l'information de résiliation."""


# --- Import ----------------------------------------------------------------


def load_events(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Charger un fichier de logs bruts (`.parquet`, `.csv` ou `.csv.gz`).

    Parameters
    ----------
    path:
        Chemin du fichier.
    columns:
        Sous-ensemble de colonnes à lire. `None` lit tout le fichier.

    Returns
    -------
    DataFrame dont les colonnes temporelles sont converties en `datetime` et
    dont `userId` est de type `str`.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    SchemaError
        Si l'extension n'est pas reconnue.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de logs introuvable : {path}")

    suffixes = {s.lower() for s in path.suffixes}
    if ".parquet" in suffixes:
        df = pd.read_parquet(path, columns=list(columns) if columns else None)
    elif ".csv" in suffixes:
        df = pd.read_csv(path, usecols=list(columns) if columns else None)
    else:
        raise SchemaError(
            f"Extension non reconnue pour {path.name} : attendu .parquet, .csv ou .csv.gz"
        )

    return coerce_types(df)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliser les types des colonnes clés.

    `userId` devient une chaîne (le dataset mélange entiers et chaînes selon
    les fichiers) et les colonnes temporelles deviennent des `datetime`.
    """
    out = df.copy()

    if "userId" in out.columns:
        out["userId"] = out["userId"].astype("string").astype(object)

    for col in DATETIME_COLUMNS:
        if col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce")

    return out


def validate_schema(
    df: pd.DataFrame, required: Sequence[str] = REQUIRED_COLUMNS
) -> pd.DataFrame:
    """Vérifier que les colonnes nécessaires au feature engineering sont là.

    Renvoie le DataFrame inchangé pour pouvoir chaîner les appels.

    Raises
    ------
    SchemaError
        Si des colonnes manquent, ou si `time` / `registration` ne sont pas
        convertibles en datetime.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            "Colonnes manquantes dans les logs : "
            + ", ".join(missing)
            + f" (colonnes reçues : {', '.join(map(str, df.columns))})"
        )

    for col in DATETIME_COLUMNS:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise SchemaError(
                f"La colonne '{col}' doit être de type datetime, reçu {df[col].dtype}"
            )

    return df


# --- Filtrage --------------------------------------------------------------


def filter_valid_users(df: pd.DataFrame) -> pd.DataFrame:
    """Retirer les événements non rattachables à un utilisateur identifié.

    Le dataset représente les sessions déconnectées par une chaîne vide dans
    `userId`, pas par une valeur manquante : un filtre sur `isna()` seul les
    laisserait passer et créerait un utilisateur fantôme agrégeant toutes ces
    sessions.
    """
    user_id = df["userId"]
    keep = user_id.notna() & (user_id.astype("string").str.strip() != "")
    return df.loc[keep].copy()


def truncate_at(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Ne garder que les événements survenus au plus tard à `cutoff`.

    C'est la brique centrale du protocole anti-fuite : tout ce qui alimente
    les features passe d'abord par ici.
    """
    return df.loc[df["time"] <= pd.Timestamp(cutoff)].copy()


def restrict_to_users(df: pd.DataFrame, users: Iterable) -> pd.DataFrame:
    """Ne garder que les événements des utilisateurs listés."""
    wanted = {str(u) for u in users}
    return df.loc[df["userId"].astype("string").isin(wanted)].copy()


def drop_leaky_pages(
    df: pd.DataFrame, leaky_pages: Sequence[str] = LEAKY_PAGES
) -> pd.DataFrame:
    """Retirer les événements de résiliation et de rétrogradation.

    Utilisé pour les comptages de pages et l'entropie, où la seule présence
    d'une page `Cancel` ou `Downgrade` suffirait à révéler l'intention.
    """
    return df.loc[~df["page"].isin(list(leaky_pages))].copy()


# --- Population et cible ---------------------------------------------------


def eligible_users(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Index:
    """Utilisateurs ayant au moins un événement avant le cutoff.

    Un utilisateur dont toute l'activité est postérieure au cutoff n'a aucun
    historique observable : on ne peut rien prédire pour lui.
    """
    return pd.Index(truncate_at(df, cutoff)["userId"].unique(), name="userId")


def churned_users(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    window_days: int,
    churn_page: str = CHURN_PAGE,
) -> pd.Index:
    """Utilisateurs ayant résilié dans la fenêtre `]cutoff, cutoff + window]`.

    La borne inférieure est stricte : une résiliation survenue exactement au
    cutoff appartient au passé observable, pas au futur à prédire.
    """
    cutoff = pd.Timestamp(cutoff)
    window_end = cutoff + pd.Timedelta(days=window_days)

    future = df.loc[(df["time"] > cutoff) & (df["time"] <= window_end)]
    churned = future.loc[future["page"] == churn_page, "userId"].unique()

    return pd.Index(churned, name="userId")


def build_target(features: pd.DataFrame, churn_users: Iterable) -> pd.DataFrame:
    """Ajouter la colonne binaire `target` au jeu de features.

    La comparaison se fait sur des chaînes des deux côtés : `userId` peut
    arriver en entier depuis un parquet et en chaîne depuis un CSV.
    """
    out = features.copy()
    churned = {str(u) for u in churn_users}
    out["target"] = out["userId"].astype("string").isin(churned).astype(int)
    return out


# --- Garde-fou anti-fuite --------------------------------------------------


def find_leaky_columns(
    columns: Iterable[str], tokens: Sequence[str] = FORBIDDEN_TOKENS
) -> list[str]:
    """Lister les colonnes dont le nom contient un jeton interdit."""
    return [c for c in columns if any(tok in str(c).lower() for tok in tokens)]


def assert_no_leaky_columns(
    columns: Iterable[str], tokens: Sequence[str] = FORBIDDEN_TOKENS
) -> None:
    """Échouer si une colonne trahit l'information de résiliation.

    Raises
    ------
    LeakageError
    """
    bad = find_leaky_columns(columns, tokens)
    if bad:
        raise LeakageError(f"Fuite détectée dans les colonnes : {bad}")


# --- Composition -----------------------------------------------------------


def prepare_events(
    path: str | Path,
    required: Sequence[str] = REQUIRED_COLUMNS,
    drop_invalid_users: bool = True,
) -> pd.DataFrame:
    """Charger, valider et nettoyer un fichier de logs en une passe.

    C'est le point d'entrée utilisé par l'application et par les scripts.
    """
    df = load_events(path)
    validate_schema(df, required=required)
    if drop_invalid_users:
        df = filter_valid_users(df)
    return df
