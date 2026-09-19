"""Construction des features utilisateur à partir des logs d'événements.

Chaque bloc `fe_*` prend le journal d'événements déjà tronqué au cutoff et
enrichit la table de features indexée par `userId`. `build_features` les
enchaîne dans un ordre qui compte : chaque bloc peut consommer des colonnes
produites par les précédents.

Protocole anti-fuite
--------------------
Deux garde-fous se superposent :

1. `build_features` tronque les événements au cutoff, donc aucun événement
   postérieur à l'instant de prédiction n'entre dans le calcul ;
2. les comptages de pages et l'entropie excluent en plus les pages de
   résiliation (`LEAKY_PAGES`), et un contrôle final rejette toute colonne
   dont le nom contient un jeton interdit.

Note : les volumétries globales (`total_page_views`, les fenêtres récentes,
les découpages horaires) comptent *tous* les événements antérieurs au cutoff,
pages de résiliation comprises. Ce n'est pas une fuite du futur — tout est
antérieur au cutoff — mais c'est un choix délibéré, hérité du notebook
d'origine, et il faut le garder en tête : un utilisateur ayant rétrogradé son
abonnement avant le cutoff contribue à ces compteurs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import entropy

from .config import LEAKY_PAGES
from .data import assert_no_leaky_columns, drop_leaky_pages, restrict_to_users, truncate_at

__all__ = [
    "build_features",
    "align_features",
    "fe_core",
    "fe_temporal",
    "fe_page_counts",
    "fe_recent",
    "fe_engagement",
    "fe_session_stats",
    "fe_inter_event_gaps",
    "fe_subscription",
    "fe_time_patterns",
    "fe_day_parts",
    "fe_entropy",
    "fe_device_and_location",
    "fe_diversity",
]


# --- Blocs élémentaires ----------------------------------------------------


def fe_core(events: pd.DataFrame) -> pd.DataFrame:
    """Agrégats de base par utilisateur : volumétrie, écoute, bornes temporelles."""
    g = events.groupby("userId")

    return g.agg(
        num_sessions=("sessionId", "nunique"),
        total_page_views=("page", "count"),
        songs_played=("song", lambda x: x.notna().sum()),
        avg_song_length=("length", "mean"),
        total_listening_time=("length", "sum"),
        std_song_length=("length", "std"),
        avg_items_per_session=("itemInSession", "mean"),
        max_items_per_session=("itemInSession", "max"),
        std_items_per_session=("itemInSession", "std"),
        first_activity=("time", "min"),
        last_activity=("time", "max"),
        unique_artists=("artist", "nunique"),
    ).reset_index()


def fe_temporal(
    events: pd.DataFrame, features: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Ancienneté, récence et volumétries normalisées par le temps.

    Les ratios `pages_per_day` et `sessions_per_day` sont importants : ils
    neutralisent en partie le fait qu'un historique tronqué est mécaniquement
    plus court qu'un historique complet.
    """
    features = features.copy()

    features["days_active"] = (
        features["last_activity"] - features["first_activity"]
    ).dt.days + 1

    reg = (
        events.groupby("userId")["registration"]
        .first()
        .reset_index()
        .rename(columns={"registration": "registration_time"})
    )
    features = features.merge(reg, on="userId", how="left")

    features["days_since_registration"] = (
        cutoff - features["registration_time"]
    ).dt.days
    features["days_since_last_activity"] = (cutoff - features["last_activity"]).dt.days

    features["pages_per_day"] = features["total_page_views"] / features[
        "days_active"
    ].clip(lower=1)
    features["sessions_per_day"] = features["num_sessions"] / features[
        "days_active"
    ].clip(lower=1)
    features["songs_per_session"] = features["songs_played"] / features[
        "num_sessions"
    ].clip(lower=1)

    features["recency_ratio"] = features["days_since_last_activity"] / features[
        "days_since_registration"
    ].clip(lower=1)

    return features


def fe_page_counts(
    events: pd.DataFrame,
    features: pd.DataFrame,
    leaky_pages: Sequence[str] = LEAKY_PAGES,
) -> pd.DataFrame:
    """Comptage par type de page, hors pages de résiliation."""
    events_no_leak = drop_leaky_pages(events, leaky_pages)

    page_counts = (
        events_no_leak.groupby(["userId", "page"]).size().unstack(fill_value=0)
    )
    page_counts.columns = ["page_" + str(c).replace(" ", "_") for c in page_counts.columns]

    features = features.merge(page_counts.reset_index(), on="userId", how="left")

    page_cols = [c for c in features.columns if c.startswith("page_")]
    features[page_cols] = features[page_cols].fillna(0)
    features["num_unique_pages"] = (features[page_cols] > 0).sum(axis=1)

    return features


def fe_recent(
    events: pd.DataFrame,
    features: pd.DataFrame,
    cutoff: pd.Timestamp,
    windows: Sequence[int] = (3, 7, 14),
) -> pd.DataFrame:
    """Activité sur des fenêtres glissantes se terminant au cutoff.

    Ces fenêtres sont fixes et ancrées sur le cutoff : elles sont donc
    comparables entre un utilisateur au long historique et un nouvel inscrit.
    """
    features = features.copy()

    for w in windows:
        sub = events.loc[events["time"] >= pd.Timestamp(cutoff) - pd.Timedelta(days=w)]
        agg = (
            sub.groupby("userId")
            .agg(pages=("page", "count"), songs=("song", lambda x: x.notna().sum()))
            .reset_index()
        )
        agg.columns = ["userId", f"recent_{w}d_pages", f"recent_{w}d_songs"]
        features = features.merge(agg, on="userId", how="left")
        features[f"recent_{w}d_pages"] = features[f"recent_{w}d_pages"].fillna(0)
        features[f"recent_{w}d_songs"] = features[f"recent_{w}d_songs"].fillna(0)

    features["activity_trend_7d"] = (features["recent_7d_pages"] / 7.0) / (
        features["pages_per_day"] + 0.01
    )
    features["activity_trend_3d"] = (features["recent_3d_pages"] / 3.0) / (
        features["pages_per_day"] + 0.01
    )

    old_pages = features["total_page_views"] - features["recent_7d_pages"]
    features["recent_vs_old_ratio"] = features["recent_7d_pages"] / (old_pages + 1)

    features["inactivity_pattern"] = (
        (features["recent_3d_pages"] == 0).astype(int)
        + (features["recent_7d_pages"] < 5).astype(int)
        + (features["recent_14d_pages"] < 10).astype(int)
    )

    features["activity_acceleration"] = (
        features["recent_3d_pages"] / 3.0 - features["recent_7d_pages"] / 7.0
    )

    return features


def fe_engagement(
    events: pd.DataFrame, features: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Intensité et qualité de l'engagement : erreurs, aide, interactions, pubs.

    Dépend de `fe_page_counts` (colonnes `page_*`) et de `fe_temporal`.
    """
    features = features.copy()

    last_song = (
        events.loc[events["song"].notna()].groupby("userId")["time"].max().rename("last_song_time")
    )
    features = features.merge(last_song, on="userId", how="left")

    features["days_without_song"] = (cutoff - features["last_song_time"]).dt.days
    features["days_without_song"] = features["days_without_song"].fillna(
        features["days_since_registration"]
    )
    features = features.drop(columns=["last_song_time"])

    features["error_rate"] = _col(features, "page_Error") / (
        features["total_page_views"] + 1
    )
    features["help_rate"] = _col(features, "page_Help") / (features["num_sessions"] + 1)

    features["session_consistency"] = 1.0 / (features["std_items_per_session"] + 1.0)

    interaction_cols = [
        "page_Thumbs_Up",
        "page_Thumbs_Down",
        "page_Add_Friend",
        "page_Add_to_Playlist",
    ]
    interaction_sum = sum(_col(features, c) for c in interaction_cols)
    features["engagement_depth"] = interaction_sum / (features["songs_played"] + 1)

    features["listening_intensity"] = features["songs_played"] / (
        features["days_active"] + 1
    )

    features["ad_rate"] = _col(features, "page_Roll_Advert") / (
        features["num_sessions"] + 1
    )
    features["ad_per_song"] = _col(features, "page_Roll_Advert") / (
        features["songs_played"] + 1
    )

    thumbs_up = _col(features, "page_Thumbs_Up")
    thumbs_down = _col(features, "page_Thumbs_Down")
    features["total_thumbs"] = thumbs_up + thumbs_down
    features["thumbs_ratio"] = thumbs_up / (thumbs_up + thumbs_down + 1)

    features["friends_per_session"] = _col(features, "page_Add_Friend") / features[
        "num_sessions"
    ].clip(lower=1)
    features["playlist_adds_per_session"] = _col(
        features, "page_Add_to_Playlist"
    ) / features["num_sessions"].clip(lower=1)

    return features


def fe_session_stats(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Distribution de la longueur des sessions et comparaison de la dernière."""
    session_sizes = (
        events.groupby(["userId", "sessionId"]).size().reset_index(name="session_length")
    )

    agg = (
        session_sizes.groupby("userId")["session_length"]
        .agg(["mean", "median", "std", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "session_len_mean",
                "median": "session_len_median",
                "std": "session_len_std",
                "max": "session_len_max",
            }
        )
    )
    features = features.merge(agg, on="userId", how="left").fillna(0)

    last_session = (
        session_sizes.sort_values(["userId", "sessionId"])
        .groupby("userId")
        .tail(1)[["userId", "session_length"]]
        .rename(columns={"session_length": "last_session_len"})
    )
    features = features.merge(last_session, on="userId", how="left").fillna(0)

    features["session_len_ratio"] = features["last_session_len"] / (
        features["session_len_mean"] + 1
    )

    return features


def fe_inter_event_gaps(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Statistiques des intervalles entre événements consécutifs."""
    events_sorted = events.sort_values(["userId", "time"]).copy()
    events_sorted["prev_time"] = events_sorted.groupby("userId")["time"].shift(1)
    events_sorted["gap"] = (
        events_sorted["time"] - events_sorted["prev_time"]
    ).dt.total_seconds()

    gaps = (
        events_sorted.groupby("userId")["gap"]
        .agg(["mean", "std", "median", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "gap_mean",
                "std": "gap_std",
                "median": "gap_median",
                "max": "gap_max",
            }
        )
    )
    features = features.merge(gaps, on="userId", how="left").fillna(0)
    features["gap_cv"] = features["gap_std"] / (features["gap_mean"] + 1)

    return features


def fe_subscription(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Niveau d'abonnement au premier et au dernier événement observé."""
    lvl = (
        events.sort_values("time")
        .groupby("userId")["level"]
        .agg(["first", "last"])
        .reset_index()
    )
    lvl.columns = ["userId", "first_level", "last_level"]

    features = features.merge(lvl, on="userId", how="left")
    features["is_paid"] = (features["last_level"] == "paid").astype(int)

    return features


def fe_time_patterns(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Répartition semaine / week-end et heures de pointe."""
    events = events.copy()
    events["dayofweek"] = events["time"].dt.dayofweek
    events["hour"] = events["time"].dt.hour

    weekend = events.loc[events["dayofweek"] >= 5].groupby("userId").size()
    weekday = events.loc[events["dayofweek"] < 5].groupby("userId").size()

    features = features.merge(weekend.rename("weekend_pages"), on="userId", how="left")
    features = features.merge(weekday.rename("weekday_pages"), on="userId", how="left")

    features["weekend_pages"] = features["weekend_pages"].fillna(0)
    features["weekday_pages"] = features["weekday_pages"].fillna(0)
    features["weekend_ratio"] = features["weekend_pages"] / (
        features["weekday_pages"] + 1
    )

    peak = events.loc[(events["hour"] >= 18) & (events["hour"] <= 23)]
    peak_pages = peak.groupby("userId").size().rename("peak_hour_pages")

    features = features.merge(peak_pages, on="userId", how="left")
    features["peak_hour_pages"] = features["peak_hour_pages"].fillna(0)
    features["peak_hour_ratio"] = features["peak_hour_pages"] / (
        features["total_page_views"] + 1
    )

    return features


def fe_day_parts(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Répartition matin / après-midi / nuit."""
    events = events.copy()
    events["hour"] = events["time"].dt.hour

    parts = {
        "morning_pages": (events["hour"] >= 6) & (events["hour"] < 12),
        "afternoon_pages": (events["hour"] >= 12) & (events["hour"] < 18),
        "night_pages": events["hour"] < 6,
    }

    for name, mask in parts.items():
        counts = events.loc[mask].groupby("userId").size().rename(name)
        features = features.merge(counts, on="userId", how="left")
        features[name] = features[name].fillna(0)

    features["morning_ratio"] = features["morning_pages"] / (
        features["total_page_views"] + 1
    )
    features["night_ratio"] = features["night_pages"] / (
        features["total_page_views"] + 1
    )

    return features


def fe_entropy(
    events: pd.DataFrame,
    features: pd.DataFrame,
    leaky_pages: Sequence[str] = LEAKY_PAGES,
) -> pd.DataFrame:
    """Entropie de la distribution des pages visitées.

    Mesure la diversité d'usage : une entropie faible signale un usage réduit
    à quelques actions répétées.
    """
    ev = drop_leaky_pages(events, leaky_pages)

    page_counts = ev.groupby(["userId", "page"]).size().reset_index(name="count")
    total = page_counts.groupby("userId")["count"].sum().rename("total_counts")
    page_counts = page_counts.merge(total, on="userId")
    page_counts["p"] = page_counts["count"] / page_counts["total_counts"]

    ent = (
        page_counts.groupby("userId")["p"]
        .apply(lambda x: entropy(x.to_numpy()))
        .rename("page_entropy")
    )

    return features.merge(ent, on="userId", how="left").fillna(0)


def fe_device_and_location(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Localisation dominante, mobilité géographique et système d'exploitation."""
    features = features.copy()

    # --- Localisation ---
    loc_counts = (
        events.groupby(["userId", "location"]).size().reset_index(name="loc_events")
    )
    total_events = (
        events.groupby("userId")["page"].count().reset_index(name="user_total_events")
    )
    loc_counts = loc_counts.merge(total_events, on="userId", how="left")

    main_loc = (
        loc_counts.sort_values(["userId", "loc_events"], ascending=[True, False])
        .groupby("userId")
        .head(1)[["userId", "loc_events", "user_total_events"]]
    )
    features = features.merge(main_loc, on="userId", how="left")

    n_locations = (
        loc_counts.groupby("userId")["location"].nunique().rename("num_locations")
    )
    features = features.merge(n_locations, on="userId", how="left")
    features["num_locations"] = features["num_locations"].fillna(0).astype(int)

    features["main_location_share"] = (
        features["loc_events"] / (features["user_total_events"] + 1)
    ).fillna(0)
    features["multi_location_user"] = (features["num_locations"] > 1).astype(int)

    features = features.drop(
        columns=["loc_events", "user_total_events"], errors="ignore"
    )

    # --- Système d'exploitation et type d'appareil ---
    ua = events.groupby("userId")["userAgent"].first().astype(str).reset_index()

    is_ios = ua["userAgent"].str.contains("iPhone|iPad", na=False, regex=True)
    is_android = ua["userAgent"].str.contains("Android", na=False)

    os_df = pd.DataFrame(
        {
            "userId": ua["userId"],
            "os_windows": ua["userAgent"].str.contains("Windows", na=False).astype(int),
            "os_mac": ua["userAgent"]
            .str.contains("Macintosh|Mac OS X", na=False, regex=True)
            .astype(int),
            "os_linux": ua["userAgent"].str.contains("Linux", na=False).astype(int),
            "device_mobile": (is_ios | is_android).astype(int),
        }
    )

    return features.merge(os_df, on="userId", how="left").fillna(0)


def fe_diversity(features: pd.DataFrame) -> pd.DataFrame:
    """Diversité du catalogue écouté."""
    features = features.copy()
    features["artist_diversity"] = features["unique_artists"] / (
        features["songs_played"] + 1
    )
    return features


# --- Orchestration ---------------------------------------------------------

#: Colonnes intermédiaires retirées avant la sortie : identifiants temporels
#: bruts et niveaux d'abonnement catégoriels déjà encodés par `is_paid`.
_DROP_COLUMNS = (
    "first_activity",
    "last_activity",
    "registration_time",
    "first_level",
    "last_level",
)


def build_features(
    events: pd.DataFrame,
    cutoff: pd.Timestamp,
    users: Iterable | None = None,
    leaky_pages: Sequence[str] = LEAKY_PAGES,
    check_leakage: bool = True,
) -> pd.DataFrame:
    """Construire la table de features, un utilisateur par ligne.

    Parameters
    ----------
    events:
        Journal d'événements brut. Il est tronqué au cutoff ici même : aucun
        appelant n'a besoin de le faire en amont.
    cutoff:
        Instant de prédiction. Tout ce qui lui est postérieur est ignoré.
    users:
        Population à conserver. `None` garde tous les utilisateurs présents
        avant le cutoff.
    leaky_pages:
        Pages exclues des comptages et de l'entropie.
    check_leakage:
        Si vrai, échoue quand une colonne produite porte un nom interdit.

    Returns
    -------
    DataFrame avec `userId` en première colonne et une colonne par feature.
    """
    events = truncate_at(events, cutoff)
    if users is not None:
        events = restrict_to_users(events, users)

    if events.empty:
        raise ValueError(
            f"Aucun événement avant le cutoff {pd.Timestamp(cutoff)} "
            "pour la population demandée."
        )

    # L'ordre compte : chaque bloc consomme des colonnes produites plus haut.
    f = fe_core(events)
    f = fe_temporal(events, f, cutoff)
    f = fe_page_counts(events, f, leaky_pages)
    f = fe_recent(events, f, cutoff)
    f = fe_engagement(events, f, cutoff)
    f = fe_session_stats(events, f)
    f = fe_inter_event_gaps(events, f)
    f = fe_subscription(events, f)
    f = fe_time_patterns(events, f)
    f = fe_day_parts(events, f)
    f = fe_entropy(events, f, leaky_pages)
    f = fe_device_and_location(events, f)
    f = fe_diversity(f)

    f = f.drop(columns=[c for c in _DROP_COLUMNS if c in f.columns])

    # Filet de sécurité : plus aucune colonne temporelle brute en sortie.
    datetime_cols = [
        c for c in f.columns if pd.api.types.is_datetime64_any_dtype(f[c])
    ]
    f = f.drop(columns=datetime_cols)

    if check_leakage:
        assert_no_leaky_columns(f.columns)

    return f.fillna(0)


def feature_columns(
    features: pd.DataFrame, exclude: Sequence[str] = ("userId", "target")
) -> list[str]:
    """Liste triée des colonnes de features, hors identifiant et cible."""
    return sorted(c for c in features.columns if c not in exclude)


def align_features(
    features: pd.DataFrame,
    columns: Sequence[str],
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Forcer le jeu de features à une liste de colonnes de référence.

    Indispensable à l'inférence : les colonnes `page_*` dépendent des pages
    effectivement présentes dans le fichier traité, donc un fichier uploadé
    produit presque toujours un jeu de colonnes différent de celui vu à
    l'entraînement. Les colonnes absentes sont créées à `fill_value`, les
    colonnes en trop sont écartées, et l'ordre est celui de `columns`.
    """
    out = features.copy()

    for col in columns:
        if col not in out.columns:
            out[col] = fill_value

    ordered = list(columns)
    extra = [c for c in ("userId", "target") if c in out.columns and c not in ordered]

    return out[extra + ordered]


def _col(features: pd.DataFrame, name: str) -> pd.Series:
    """Renvoyer une colonne, ou une colonne de zéros si elle est absente.

    Les colonnes `page_*` n'existent que si la page correspondante apparaît
    dans les données traitées ; sur un petit échantillon, beaucoup manquent.
    """
    if name in features.columns:
        return features[name]
    return pd.Series(np.zeros(len(features)), index=features.index, name=name)
