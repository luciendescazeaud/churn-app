"""Chargement des artefacts et prédiction sur de nouveaux journaux.

C'est la couche que l'application appelle. Elle ne connaît ni les données
d'entraînement ni la façon dont le modèle a été obtenu : elle lit quatre
fichiers et applique le même feature engineering qu'à l'entraînement.

Le seuil est chargé depuis les artefacts et non redéfini ici. C'est ce qui
garantit que les prédictions affichées correspondent aux métriques annoncées.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .config import SplitConfig
from .data import filter_valid_users, validate_schema
from .features import align_features, build_features

__all__ = ["Artifacts", "load_artifacts", "predict_from_events"]


class ArtifactsError(FileNotFoundError):
    """Les artefacts du modèle sont absents ou incomplets."""


@dataclass(frozen=True)
class Artifacts:
    """Modèle entraîné et tout ce qui l'accompagne."""

    model: object
    columns: list[str]
    threshold: float
    metrics: dict
    importance: pd.DataFrame

    @property
    def trained_at(self) -> str | None:
        return self.metrics.get("provenance", {}).get("trained_at")

    @property
    def git_commit(self) -> str | None:
        return self.metrics.get("provenance", {}).get("git_commit")


def load_artifacts(directory: str | Path) -> Artifacts:
    """Charger les artefacts produits par `churn.train`.

    Raises
    ------
    ArtifactsError
        Si le dossier ou l'un des fichiers attendus est absent. Le message
        nomme le fichier manquant et rappelle la commande qui le produit.
    """
    chemin = Path(directory)

    attendus = {
        "model": chemin / "model.joblib",
        "columns": chemin / "feature_columns.json",
        "metrics": chemin / "metrics.json",
    }

    manquants = [f.name for f in attendus.values() if not f.exists()]
    if manquants:
        raise ArtifactsError(
            f"Artefacts incomplets dans {chemin} — fichiers manquants : "
            f"{', '.join(manquants)}. Lance `python -m churn.train --synthetic "
            f"--out {chemin}` pour les produire."
        )

    metrics = json.loads(attendus["metrics"].read_text(encoding="utf-8"))
    colonnes = json.loads(attendus["columns"].read_text(encoding="utf-8"))

    fichier_importance = chemin / "feature_importance.csv"
    importance = (
        pd.read_csv(fichier_importance)
        if fichier_importance.exists()
        else pd.DataFrame(columns=["feature", "importance"])
    )

    return Artifacts(
        model=joblib.load(attendus["model"]),
        columns=colonnes,
        threshold=float(metrics["threshold"]),
        metrics=metrics,
        importance=importance,
    )


def predict_from_events(
    events: pd.DataFrame,
    artifacts: Artifacts,
    cutoff: pd.Timestamp | None = None,
    config: SplitConfig | None = None,
) -> pd.DataFrame:
    """Prédire la probabilité de résiliation de chaque utilisateur d'un journal.

    Le chemin est exactement celui de l'entraînement — validation, nettoyage,
    feature engineering, alignement des colonnes — ce qui est la seule façon
    de garantir que le modèle reçoit ce qu'il attend.

    Parameters
    ----------
    events:
        Journal brut, au format des logs d'origine.
    artifacts:
        Modèle et métadonnées chargés par `load_artifacts`.
    cutoff:
        Instant de prédiction. Par défaut, le dernier événement du journal :
        on prédit à partir de tout ce qui est observable.
    config:
        Utilisé seulement si `cutoff` est absent et que le journal est vide.

    Returns
    -------
    DataFrame `userId`, `probability`, `prediction`, trié par probabilité
    décroissante.
    """
    validate_schema(events)
    propres = filter_valid_users(events)

    if propres.empty:
        raise ValueError(
            "Aucun utilisateur identifié dans ce journal : toutes les lignes "
            "portent un userId vide."
        )

    if cutoff is None:
        cutoff = propres["time"].max()

    table = build_features(propres, cutoff=cutoff)
    alignees = align_features(table, artifacts.columns)

    X = alignees[artifacts.columns].astype(float)
    probabilites = artifacts.model.predict_proba(X)[:, 1]

    return (
        pd.DataFrame(
            {
                "userId": alignees["userId"].to_numpy(),
                "probability": probabilites,
                "prediction": (probabilites >= artifacts.threshold).astype(int),
            }
        )
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
