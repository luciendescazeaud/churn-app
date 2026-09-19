"""Prédiction de churn à partir de logs d'événements.

Point d'entrée du package : les constantes du dataset et le protocole
temporel sont dans `config`, l'import et le filtrage dans `data`, le feature
engineering dans `features`.
"""

from __future__ import annotations

from .config import (
    CHURN_PAGE,
    FORBIDDEN_TOKENS,
    LEAKY_PAGES,
    RAW_COLUMNS,
    REQUIRED_COLUMNS,
    SplitConfig,
)
from .data import (
    LeakageError,
    SchemaError,
    assert_no_leaky_columns,
    build_target,
    churned_users,
    drop_leaky_pages,
    eligible_users,
    filter_valid_users,
    find_leaky_columns,
    load_events,
    prepare_events,
    restrict_to_users,
    truncate_at,
    validate_schema,
)
from .features import align_features, build_features, feature_columns

__version__ = "0.1.0"

__all__ = [
    "CHURN_PAGE",
    "FORBIDDEN_TOKENS",
    "LEAKY_PAGES",
    "RAW_COLUMNS",
    "REQUIRED_COLUMNS",
    "SplitConfig",
    "LeakageError",
    "SchemaError",
    "align_features",
    "assert_no_leaky_columns",
    "build_features",
    "build_target",
    "churned_users",
    "drop_leaky_pages",
    "eligible_users",
    "feature_columns",
    "filter_valid_users",
    "find_leaky_columns",
    "load_events",
    "prepare_events",
    "restrict_to_users",
    "truncate_at",
    "validate_schema",
]
