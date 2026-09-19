"""Constantes du dataset et configuration du découpage temporel.

Le protocole anti-fuite du projet repose sur une seule idée : les features d'un
utilisateur ne sont construites que sur ses événements antérieurs à un instant
de coupure (`cutoff`), et la cible est observée dans la fenêtre qui suit cet
instant. Rien de ce qui se passe après le cutoff n'entre dans les features.

`SplitConfig` rend ce protocole explicite et paramétrable, là où le notebook
d'origine utilisait trois constantes globales.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# --- Schéma des logs bruts -------------------------------------------------

#: Colonnes présentes dans les fichiers `train.parquet` / `test.parquet`.
RAW_COLUMNS: tuple[str, ...] = (
    "status",
    "gender",
    "firstName",
    "level",
    "lastName",
    "userId",
    "ts",
    "auth",
    "page",
    "sessionId",
    "location",
    "itemInSession",
    "userAgent",
    "method",
    "length",
    "song",
    "artist",
    "time",
    "registration",
)

#: Sous-ensemble réellement consommé par le feature engineering. C'est ce qui
#: est exigé d'un fichier uploadé dans l'application.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "userId",
    "sessionId",
    "page",
    "time",
    "registration",
    "itemInSession",
    "length",
    "song",
    "artist",
    "level",
    "location",
    "userAgent",
)

#: Colonnes à convertir en datetime à l'import.
DATETIME_COLUMNS: tuple[str, ...] = ("time", "registration")

# --- Définition du churn ---------------------------------------------------

#: Événement qui matérialise le churn d'un utilisateur.
CHURN_PAGE = "Cancellation Confirmation"

#: Pages exclues des comptages de pages et de l'entropie : elles décrivent
#: l'acte de résiliation lui-même ou son antichambre immédiate.
LEAKY_PAGES: tuple[str, ...] = (
    "Cancellation Confirmation",
    "Cancel",
    "Downgrade",
    "Submit Downgrade",
    "Submit Cancellation",
)

#: Jetons interdits dans les noms de colonnes du jeu de features final.
#: Sert de garde-fou : toute feature dont le nom contient l'un d'eux trahit
#: une fuite de l'information de résiliation.
FORBIDDEN_TOKENS: tuple[str, ...] = (
    "cancel",
    "cancellation",
    "downgrade",
    "submit_downgrade",
    "submit_cancellation",
)

# --- Découpage temporel ----------------------------------------------------

#: Fin de la période d'observation du jeu de test de la compétition.
DEFAULT_FINAL_CUTOFF = pd.Timestamp("2018-11-20")

#: Horizon de prédiction, en jours.
DEFAULT_CHURN_WINDOW_DAYS = 10


@dataclass(frozen=True)
class SplitConfig:
    """Protocole temporel d'entraînement et d'inférence.

    Attributes
    ----------
    final_cutoff:
        Instant de coupure du jeu de test : les features de test sont
        construites sur tout ce qui précède cette date.
    churn_window_days:
        Horizon de prédiction. On cherche à prédire une résiliation survenant
        dans les `churn_window_days` jours suivant le cutoff.

    Notes
    -----
    Le cutoff d'entraînement est reculé d'exactement une fenêtre de churn, de
    sorte que la fenêtre d'observation de la cible d'entraînement se termine
    au cutoff de test. Entraînement et inférence voient donc la même longueur
    d'horizon, et aucun événement postérieur au cutoff d'entraînement
    n'alimente les features d'entraînement.

    >>> cfg = SplitConfig()
    >>> cfg.train_cutoff
    Timestamp('2018-11-10 00:00:00')
    >>> cfg.train_churn_end == cfg.final_cutoff
    True
    """

    final_cutoff: pd.Timestamp = DEFAULT_FINAL_CUTOFF
    churn_window_days: int = DEFAULT_CHURN_WINDOW_DAYS

    def __post_init__(self) -> None:
        if self.churn_window_days <= 0:
            raise ValueError(
                f"churn_window_days doit être strictement positif, reçu {self.churn_window_days}"
            )
        # `frozen=True` interdit l'affectation directe.
        object.__setattr__(self, "final_cutoff", pd.Timestamp(self.final_cutoff))

    @property
    def churn_window(self) -> pd.Timedelta:
        return pd.Timedelta(days=self.churn_window_days)

    @property
    def train_cutoff(self) -> pd.Timestamp:
        """Instant de coupure des features d'entraînement."""
        return self.final_cutoff - self.churn_window

    @property
    def train_churn_end(self) -> pd.Timestamp:
        """Fin de la fenêtre d'observation de la cible d'entraînement."""
        return self.train_cutoff + self.churn_window
