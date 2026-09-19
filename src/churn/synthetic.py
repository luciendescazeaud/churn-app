"""Génération de logs d'événements synthétiques.

Ce module produit des journaux au format exact des données brutes, sans en
reprendre aucun contenu réel. Il sert deux usages qui doivent rester
cohérents entre eux :

* les **fixtures de tests**, en appelant `make_events` directement ;
* la **démonstration de l'application**, via le bouton « charger un jeu
  d'exemple », qui appelle la même fonction.

Faire des deux le produit d'une seule fonction garantit que ce qui est testé
est exactement ce que l'application manipule. Le tirage est entièrement
déterminé par `seed` : deux appels avec la même graine produisent le même
DataFrame, ligne pour ligne.

Aucune donnée de compétition n'est redistribuée : tout est tiré ici.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CHURN_PAGE, RAW_COLUMNS, SplitConfig

__all__ = ["make_events", "write_demo_csv", "PAGE_WEIGHTS"]


#: Distribution des pages, grossièrement calquée sur celle d'un service de
#: streaming musical : l'écoute domine, les actions sociales sont rares.
PAGE_WEIGHTS: dict[str, float] = {
    "NextSong": 0.72,
    "Thumbs Up": 0.05,
    "Home": 0.05,
    "Add to Playlist": 0.03,
    "Roll Advert": 0.04,
    "Add Friend": 0.02,
    "Thumbs Down": 0.01,
    "Help": 0.01,
    "Settings": 0.01,
    "Save Settings": 0.01,
    "About": 0.01,
    "Logout": 0.02,
    "Error": 0.005,
    "Upgrade": 0.005,
    "Downgrade": 0.01,
}

_USER_AGENTS = (
    '"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"',
    '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/605.1.15"',
    '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"',
    '"Mozilla/5.0 (iPhone; CPU iPhone OS 12_4 like Mac OS X)"',
    '"Mozilla/5.0 (Linux; Android 9) AppleWebKit/537.36"',
)

_LOCATIONS = (
    "Paris, Île-de-France",
    "Lyon, Auvergne-Rhône-Alpes",
    "Marseille, Provence-Alpes-Côte d'Azur",
    "Bordeaux, Nouvelle-Aquitaine",
    "Lille, Hauts-de-France",
    "Toulouse, Occitanie",
)

_FIRST_NAMES = ("Camille", "Louis", "Inès", "Hugo", "Léa", "Noah", "Jade", "Sacha")
_LAST_NAMES = ("Martin", "Bernard", "Dubois", "Moreau", "Laurent", "Garnier")


def make_events(
    n_users: int = 200,
    seed: int = 0,
    config: SplitConfig | None = None,
    churn_rate: float = 0.12,
    history_days: int = 60,
    logged_out_events: int = 60,
) -> pd.DataFrame:
    """Produire un journal d'événements synthétique.

    Parameters
    ----------
    n_users:
        Nombre d'utilisateurs identifiés à générer.
    seed:
        Graine du tirage. Même graine, même résultat.
    config:
        Protocole temporel. La fenêtre d'observation se termine au
        `final_cutoff`, et les résiliations tombent entre le `train_cutoff` et
        le `final_cutoff`, c'est-à-dire exactement dans la fenêtre que le
        modèle doit apprendre à anticiper.
    churn_rate:
        Proportion d'utilisateurs qui résilient.
    history_days:
        Profondeur de l'historique avant le `final_cutoff`.
    logged_out_events:
        Nombre d'événements de session déconnectée, portant un `userId` vide.
        Ils reproduisent un défaut réel des données et donnent de la matière
        aux tests de `filter_valid_users`.

    Returns
    -------
    DataFrame aux colonnes de `RAW_COLUMNS`, trié par `time`.
    """
    if not 0.0 <= churn_rate <= 1.0:
        raise ValueError(f"churn_rate doit être dans [0, 1], reçu {churn_rate}")
    if n_users <= 0:
        raise ValueError(f"n_users doit être strictement positif, reçu {n_users}")

    cfg = config or SplitConfig()
    rng = np.random.default_rng(seed)

    window_end = cfg.final_cutoff
    window_start = window_end - pd.Timedelta(days=history_days)

    pages = np.array(list(PAGE_WEIGHTS))
    weights = np.array(list(PAGE_WEIGHTS.values()))
    weights = weights / weights.sum()

    n_churners = int(round(n_users * churn_rate))
    churner_flags = np.zeros(n_users, dtype=bool)
    churner_flags[:n_churners] = True
    rng.shuffle(churner_flags)

    records: list[dict] = []

    for i in range(n_users):
        user_id = str(1000 + i)
        is_churner = bool(churner_flags[i])

        registration = window_start - pd.Timedelta(
            days=float(rng.uniform(10, 400)), seconds=float(rng.integers(0, 86400))
        )

        # Un churner cesse toute activité à sa résiliation, qui tombe dans la
        # fenêtre de prédiction. Les autres restent actifs jusqu'au bout.
        if is_churner:
            cancel_time = cfg.train_cutoff + pd.Timedelta(
                seconds=float(rng.uniform(1, cfg.churn_window.total_seconds()))
            )
            activity_end = cancel_time
        else:
            cancel_time = None
            activity_end = window_end

        activity_start = max(window_start, registration)
        span = (activity_end - activity_start).total_seconds()
        if span <= 3600:
            continue

        level = "paid" if rng.random() < 0.65 else "free"
        user_agent = _USER_AGENTS[int(rng.integers(0, len(_USER_AGENTS)))]
        home = _LOCATIONS[int(rng.integers(0, len(_LOCATIONS)))]
        gender = "F" if rng.random() < 0.5 else "M"
        first_name = _FIRST_NAMES[int(rng.integers(0, len(_FIRST_NAMES)))]
        last_name = _LAST_NAMES[int(rng.integers(0, len(_LAST_NAMES)))]

        n_sessions = int(rng.integers(2, 26))
        session_starts = np.sort(rng.uniform(0, span, n_sessions))

        for s, offset in enumerate(session_starts):
            session_id = i * 1000 + s
            session_start = activity_start + pd.Timedelta(seconds=float(offset))
            n_events = int(rng.integers(2, 45))

            elapsed = 0.0
            for item in range(n_events):
                page = str(rng.choice(pages, p=weights))
                is_song = page == "NextSong"
                length = float(rng.normal(245, 95)) if is_song else np.nan
                if is_song:
                    length = max(30.0, length)

                event_time = session_start + pd.Timedelta(seconds=elapsed)
                if event_time > activity_end:
                    break
                elapsed += float(length if is_song else rng.uniform(5, 60))

                # Un utilisateur change parfois de lieu de connexion.
                location = (
                    home
                    if rng.random() < 0.9
                    else _LOCATIONS[int(rng.integers(0, len(_LOCATIONS)))]
                )

                records.append(
                    _record(
                        user_id=user_id,
                        session_id=session_id,
                        item=item,
                        page=page,
                        time=event_time,
                        registration=registration,
                        length=length,
                        song=f"song_{int(rng.integers(0, 4000))}" if is_song else None,
                        artist=f"artist_{int(rng.integers(0, 400))}" if is_song else None,
                        level=level,
                        location=location,
                        user_agent=user_agent,
                        gender=gender,
                        first_name=first_name,
                        last_name=last_name,
                        auth="Logged In",
                    )
                )

        if cancel_time is not None:
            records.append(
                _record(
                    user_id=user_id,
                    session_id=i * 1000 + n_sessions,
                    item=0,
                    page=CHURN_PAGE,
                    time=cancel_time,
                    registration=registration,
                    length=np.nan,
                    song=None,
                    artist=None,
                    level=level,
                    location=home,
                    user_agent=user_agent,
                    gender=gender,
                    first_name=first_name,
                    last_name=last_name,
                    auth="Cancelled",
                )
            )

    # Sessions déconnectées : `userId` vide, pas manquant. C'est ainsi que le
    # défaut se présente dans les données réelles.
    for k in range(logged_out_events):
        records.append(
            _record(
                user_id="",
                session_id=900_000 + k,
                item=0,
                page="Home",
                time=window_start
                + pd.Timedelta(seconds=float(rng.uniform(0, history_days * 86400))),
                registration=pd.NaT,
                length=np.nan,
                song=None,
                artist=None,
                level="free",
                location=_LOCATIONS[int(rng.integers(0, len(_LOCATIONS)))],
                user_agent=_USER_AGENTS[int(rng.integers(0, len(_USER_AGENTS)))],
                gender=None,
                first_name=None,
                last_name=None,
                auth="Logged Out",
            )
        )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    df["ts"] = (df["time"].astype("int64") // 1_000_000).astype("int64")

    return df[list(RAW_COLUMNS)]


def _record(
    *,
    user_id: str,
    session_id: int,
    item: int,
    page: str,
    time: pd.Timestamp,
    registration: pd.Timestamp,
    length: float,
    song: str | None,
    artist: str | None,
    level: str,
    location: str,
    user_agent: str,
    gender: str | None,
    first_name: str | None,
    last_name: str | None,
    auth: str,
) -> dict:
    """Assembler une ligne au format des logs bruts."""
    return {
        "status": 200,
        "gender": gender,
        "firstName": first_name,
        "level": level,
        "lastName": last_name,
        "userId": user_id,
        "ts": 0,  # recalculé après le tri
        "auth": auth,
        "page": page,
        "sessionId": session_id,
        "location": location,
        "itemInSession": item,
        "userAgent": user_agent,
        "method": "PUT" if page == "NextSong" else "GET",
        "length": length,
        "song": song,
        "artist": artist,
        "time": time,
        "registration": registration,
    }


def write_demo_csv(path: str | Path, **kwargs) -> Path:
    """Écrire un journal synthétique dans un fichier CSV.

    Le fichier produit sert de jeu d'exemple téléchargeable et de cible pour
    tester le chemin d'import de l'application.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_events(**kwargs).to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Générer un journal d'événements synthétique."
    )
    parser.add_argument("--out", default="demo_events.csv", help="Fichier CSV de sortie")
    parser.add_argument("--users", type=int, default=200, help="Nombre d'utilisateurs")
    parser.add_argument("--seed", type=int, default=0, help="Graine du tirage")
    parser.add_argument(
        "--churn-rate", type=float, default=0.12, help="Proportion de résiliations"
    )
    parser.add_argument(
        "--history-days", type=int, default=60, help="Profondeur de l'historique"
    )
    args = parser.parse_args(argv)

    path = write_demo_csv(
        args.out,
        n_users=args.users,
        seed=args.seed,
        churn_rate=args.churn_rate,
        history_days=args.history_days,
    )
    size_kb = path.stat().st_size / 1024
    print(f"Écrit : {path} ({size_kb:.0f} Ko)")


if __name__ == "__main__":
    main()
