"""Choix du seuil de décision et calcul des métriques.

Le modèle produit une probabilité ; la décision « cet utilisateur va résilier »
suppose un seuil. Avec une cible à 3,5 % de positifs, le seuil par défaut de
0,5 classerait presque tout le monde en non-résiliant et donnerait une
balanced accuracy proche de 0,5 — inutile.

Le seuil est donc un **paramètre appris**, au même titre que les poids du
modèle. Il doit être choisi sur des prédictions que le modèle n'a pas vues, et
sauvegardé avec lui : sans cela, l'application ne reproduit pas les métriques
annoncées.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "scan_thresholds",
    "best_threshold",
    "classification_metrics",
]


def scan_thresholds(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    n_thresholds: int = 400,
) -> pd.DataFrame:
    """Balayer les seuils et mesurer la balanced accuracy de chacun.

    Les seuils candidats sont tirés des quantiles des probabilités prédites
    plutôt que d'une grille fixe : la plage utile dépend du modèle et de la
    prévalence, et une grille arbitraire sur [0, 1] passerait à côté de
    l'optimum quand les probabilités sont toutes petites.

    Returns
    -------
    DataFrame trié par seuil, avec la balanced accuracy, les deux rappels et
    la proportion de positifs prédits.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    quantiles = np.linspace(0.0, 1.0, n_thresholds)
    candidats = np.unique(np.quantile(proba, quantiles))

    lignes = []
    for seuil in candidats:
        preds = (proba >= seuil).astype(int)
        matrice = confusion_matrix(y_true, preds, labels=[0, 1])
        vn, fp, fn, vp = matrice.ravel()

        lignes.append(
            {
                "threshold": float(seuil),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
                "recall_churn": float(vp / (vp + fn)) if (vp + fn) else 0.0,
                "recall_non_churn": float(vn / (vn + fp)) if (vn + fp) else 0.0,
                "predicted_positive_rate": float(preds.mean()),
            }
        )

    return pd.DataFrame(lignes)


def best_threshold(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    n_thresholds: int = 400,
) -> tuple[float, float]:
    """Seuil maximisant la balanced accuracy, et la valeur atteinte.

    En cas d'égalité — fréquente, la balanced accuracy étant constante par
    morceaux — le seuil le plus élevé est retenu : à performance égale, il
    prédit moins de positifs, donc déclenche moins d'actions commerciales.
    """
    table = scan_thresholds(y_true, proba, n_thresholds)
    meilleur = table["balanced_accuracy"].max()
    candidats = table.loc[table["balanced_accuracy"] == meilleur, "threshold"]

    return float(candidats.max()), float(meilleur)


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    threshold: float,
) -> dict:
    """Métriques complètes à un seuil donné.

    L'AUC est indépendante du seuil et mesure la qualité du classement ; la
    balanced accuracy mesure la qualité de la décision. Les deux sont
    rapportées, car un bon classement mal seuillé donne de mauvaises
    décisions.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    preds = (proba >= threshold).astype(int)

    vn, fp, fn, vp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "recall_churn": float(recall_score(y_true, preds, zero_division=0)),
        "recall_non_churn": float(vn / (vn + fp)) if (vn + fp) else 0.0,
        "precision_churn": float(precision_score(y_true, preds, zero_division=0)),
        "predicted_positive_rate": float(preds.mean()),
        "actual_positive_rate": float(y_true.mean()),
        "confusion_matrix": {
            "true_negative": int(vn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(vp),
        },
        "n_samples": int(len(y_true)),
    }
