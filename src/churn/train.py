"""Entraînement du modèle et production des artefacts.

Le pipeline complet, exécutable depuis le terminal :

    python -m churn.train --events data/train.parquet --out artifacts
    python -m churn.train --synthetic --out artifacts/demo

Il produit quatre fichiers dans le dossier de sortie :

* ``model.joblib`` — le modèle entraîné ;
* ``feature_columns.json`` — l'ordre exact des colonnes attendues ;
* ``metrics.json`` — seuil, métriques et provenance ;
* ``feature_importance.csv`` — contributions des features.

Ce sont ces quatre fichiers, et eux seuls, que l'application charge. Elle n'a
jamais besoin des données d'entraînement.

Choix méthodologique
--------------------
Le seuil de décision est choisi sur des **prédictions hors échantillon**,
obtenues par validation croisée : chaque utilisateur est prédit par un modèle
qui ne l'a pas vu à l'entraînement. C'est indispensable ici. Choisir le seuil
sur le même jeu que celui qui a servi à l'arrêt anticipé donnerait une
balanced accuracy optimiste sur deux plans à la fois. Et avec 3,5 % de
positifs, une simple découpe à 20 % ne laisserait qu'une poignée de
résiliations pour calibrer le seuil — beaucoup trop peu pour être stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import SplitConfig
from .data import build_target, churned_users, eligible_users, filter_valid_users, load_events
from .evaluation import best_threshold, classification_metrics, scan_thresholds
from .features import build_features, feature_columns

__all__ = ["TrainingResult", "train_from_events", "save_artifacts", "DEFAULT_PARAMS"]


#: Hyperparamètres LightGBM. Repris du notebook, à deux exceptions près :
#: `n_estimators` est un plafond que l'arrêt anticipé ne laisse jamais
#: atteindre, et `verbose=-1` fait taire les avertissements de compatibilité.
DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "learning_rate": 0.02,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_estimators": 5000,
    "n_jobs": -1,
    "verbose": -1,
}

SEED = 42


@dataclass
class TrainingResult:
    """Tout ce que produit un entraînement, avant écriture sur disque."""

    model: lgb.LGBMClassifier
    columns: list[str]
    threshold: float
    metrics: dict
    importance: pd.DataFrame
    threshold_scan: pd.DataFrame
    provenance: dict = field(default_factory=dict)


# --- Entraînement ----------------------------------------------------------


def cross_val_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_splits: int = 5,
    seed: int = SEED,
) -> tuple[np.ndarray, list[int]]:
    """Probabilités hors échantillon par validation croisée stratifiée.

    Chaque ligne reçoit une probabilité produite par un modèle entraîné sans
    elle. Les découpes sont stratifiées sur la cible, faute de quoi certains
    plis pourraient ne contenir aucune résiliation.

    Returns
    -------
    Les probabilités hors échantillon, et le nombre d'arbres retenu par
    l'arrêt anticipé dans chaque pli.
    """
    params = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed}

    oof = np.zeros(len(X), dtype=float)
    best_iterations: list[int] = []

    decoupe = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for pli, (idx_train, idx_valid) in enumerate(decoupe.split(X, y), start=1):
        modele = lgb.LGBMClassifier(**params)
        modele.fit(
            X.iloc[idx_train],
            y.iloc[idx_train],
            eval_X=X.iloc[idx_valid],
            eval_y=y.iloc[idx_valid],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )

        oof[idx_valid] = modele.predict_proba(X.iloc[idx_valid])[:, 1]
        best_iterations.append(int(modele.best_iteration_ or params["n_estimators"]))

        print(f"  pli {pli}/{n_splits} — {best_iterations[-1]} arbres")

    return oof, best_iterations


def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
    params: dict | None = None,
    seed: int = SEED,
) -> lgb.LGBMClassifier:
    """Modèle final, entraîné sur toutes les données.

    Le nombre d'arbres n'est plus déterminé par arrêt anticipé — il n'y a plus
    de jeu de validation disponible — mais fixé à la moyenne des plis.
    """
    params = {
        **DEFAULT_PARAMS,
        **(params or {}),
        "n_estimators": n_estimators,
        "random_state": seed,
    }

    modele = lgb.LGBMClassifier(**params)
    modele.fit(X, y)

    return modele


def train_from_events(
    events: pd.DataFrame,
    config: SplitConfig | None = None,
    params: dict | None = None,
    n_splits: int = 5,
    seed: int = SEED,
) -> TrainingResult:
    """Chaîne complète : journal brut → modèle entraîné et métriques.

    Les features sont construites au `train_cutoff` et la cible observée dans
    la fenêtre qui suit, conformément au protocole de `SplitConfig`.
    """
    cfg = config or SplitConfig()

    print("Nettoyage et construction de la cible…")
    propres = filter_valid_users(events)
    population = eligible_users(propres, cfg.train_cutoff)
    resiliants = churned_users(propres, cfg.train_cutoff, cfg.churn_window_days)

    print(f"  {len(population)} utilisateurs éligibles, {len(resiliants)} résiliations")

    print("Construction des features…")
    table = build_features(propres, cutoff=cfg.train_cutoff, users=population)
    table = build_target(table, resiliants)

    colonnes = feature_columns(table)
    X = table[colonnes].astype(float)
    y = table["target"].astype(int)

    if y.sum() < n_splits:
        raise ValueError(
            f"Trop peu de résiliations ({int(y.sum())}) pour {n_splits} plis de "
            "validation croisée. Élargis la fenêtre ou la population."
        )

    print(f"  {len(colonnes)} features, {y.mean():.2%} de résiliations")

    print(f"Validation croisée sur {n_splits} plis…")
    oof, best_iterations = cross_val_probabilities(X, y, params, n_splits, seed)

    seuil, ba = best_threshold(y, oof)
    print(f"  seuil retenu : {seuil:.4f} — balanced accuracy {ba:.4f}")

    metriques = classification_metrics(y, oof, seuil)

    n_arbres = int(round(float(np.mean(best_iterations))))
    print(f"Modèle final sur l'ensemble des données ({n_arbres} arbres)…")
    modele = fit_final_model(X, y, n_arbres, params, seed)

    importance = (
        pd.DataFrame(
            {"feature": colonnes, "importance": modele.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return TrainingResult(
        model=modele,
        columns=colonnes,
        threshold=seuil,
        metrics=metriques,
        importance=importance,
        threshold_scan=scan_thresholds(y, oof),
        provenance=_provenance(
            cfg=cfg,
            X=X,
            y=y,
            seed=seed,
            n_splits=n_splits,
            n_estimators=n_arbres,
            best_iterations=best_iterations,
        ),
    )


# --- Provenance ------------------------------------------------------------


def _git_commit() -> str | None:
    """Identifiant du commit courant, si le code tourne dans un dépôt Git."""
    try:
        sortie = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return sortie.stdout.strip() or None


def _dataset_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    """Empreinte du jeu d'entraînement.

    Permet de vérifier, des mois plus tard, qu'un modèle sauvegardé a bien été
    entraîné sur les données qu'on croit.
    """
    empreinte = pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes()
    empreinte += pd.util.hash_pandas_object(y, index=False).to_numpy().tobytes()

    return hashlib.sha256(empreinte).hexdigest()[:16]


def _provenance(
    *,
    cfg: SplitConfig,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_splits: int,
    n_estimators: int,
    best_iterations: list[int],
) -> dict:
    """Tout ce qu'il faut pour reproduire cet entraînement à l'identique."""
    return {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "dataset_fingerprint": _dataset_fingerprint(X, y),
        "n_users": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_churners": int(y.sum()),
        "seed": seed,
        "n_splits": n_splits,
        "n_estimators": n_estimators,
        "best_iterations_per_fold": best_iterations,
        "final_cutoff": cfg.final_cutoff.isoformat(),
        "train_cutoff": cfg.train_cutoff.isoformat(),
        "churn_window_days": cfg.churn_window_days,
        "python": platform.python_version(),
        "versions": {
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


# --- Écriture des artefacts ------------------------------------------------


def save_artifacts(result: TrainingResult, out_dir: str | Path) -> Path:
    """Écrire les quatre fichiers que l'application consommera."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.model, out / "model.joblib")

    (out / "feature_columns.json").write_text(
        json.dumps(result.columns, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (out / "metrics.json").write_text(
        json.dumps(
            {
                "threshold": result.threshold,
                "metrics": result.metrics,
                "provenance": result.provenance,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result.importance.to_csv(out / "feature_importance.csv", index=False)
    result.threshold_scan.to_csv(out / "threshold_scan.csv", index=False)

    return out


# --- Ligne de commande -----------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Entraîner le modèle de churn et produire les artefacts."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--events", help="Fichier de logs bruts (.parquet ou .csv)")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Entraîner sur des données synthétiques, sans fichier externe",
    )
    parser.add_argument("--out", default="artifacts", help="Dossier de sortie")
    parser.add_argument("--users", type=int, default=3000, help="Utilisateurs synthétiques")
    parser.add_argument("--splits", type=int, default=5, help="Plis de validation croisée")
    parser.add_argument("--seed", type=int, default=SEED, help="Graine")
    args = parser.parse_args(argv)

    if args.synthetic:
        from .synthetic import make_events

        print(f"Génération de {args.users} utilisateurs synthétiques…")
        events = make_events(n_users=args.users, seed=args.seed)
    else:
        print(f"Lecture de {args.events}…")
        events = load_events(args.events)

    resultat = train_from_events(events, n_splits=args.splits, seed=args.seed)
    chemin = save_artifacts(resultat, args.out)

    m = resultat.metrics
    print(f"\nArtefacts écrits dans {chemin}/")
    print(f"  balanced accuracy   {m['balanced_accuracy']:.4f}")
    print(f"  AUC                 {m['roc_auc']:.4f}")
    print(f"  rappel résiliants   {m['recall_churn']:.4f}")
    print(f"  rappel fidèles      {m['recall_non_churn']:.4f}")
    print(f"  seuil               {m['threshold']:.4f}")
    print(
        f"  positifs prédits    {m['predicted_positive_rate']:.2%} "
        f"(réel {m['actual_positive_rate']:.2%})"
    )


if __name__ == "__main__":
    main()
