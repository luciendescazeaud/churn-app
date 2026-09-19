"""Fixtures partagées par la suite de tests.

Deux familles de données coexistent ici, et elles ne servent pas la même chose.

Les **journaux construits à la main** (`events`) décrivent trois ou quatre
événements dont on connaît le résultat attendu par le calcul mental. Ils
servent à vérifier des règles précises : telle ligne est retenue, telle autre
écartée. Quand un tel test échoue, le diagnostic est immédiat.

Le **journal synthétique** (`synthetic_events`) est volumineux et réaliste. Il
sert aux tests d'intégration, là où l'on veut savoir si l'ensemble du pipeline
tient debout sur des données variées.

Aucune donnée de compétition n'est utilisée : la suite tourne partout, y
compris en intégration continue.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn import SplitConfig
from churn.config import RAW_COLUMNS
from churn.synthetic import make_events

#: Instant de coupure utilisé par les tests. Une date ronde, pour que les
#: décalages en jours restent lisibles dans les assertions.
CUTOFF = pd.Timestamp("2018-11-10")


@pytest.fixture
def config() -> SplitConfig:
    """Protocole temporel des tests : coupure au 10/11, fenêtre de 10 jours."""
    return SplitConfig(final_cutoff=pd.Timestamp("2018-11-20"), churn_window_days=10)


@pytest.fixture
def events():
    """Fabrique de journaux d'événements minimaux.

    Chaque appel prend une liste de dictionnaires partiels et complète les
    champs manquants par des valeurs par défaut plausibles. Un test n'a donc à
    écrire que ce qui le concerne :

    >>> events([{"userId": "1", "page": "NextSong", "time": CUTOFF}])

    Returns
    -------
    Callable[[list[dict]], pd.DataFrame]
    """

    def _make(rows: list[dict]) -> pd.DataFrame:
        defaults = {
            "status": 200,
            "gender": "F",
            "firstName": "Camille",
            "level": "paid",
            "lastName": "Martin",
            "userId": "1",
            "ts": 0,
            "auth": "Logged In",
            "page": "NextSong",
            "sessionId": 1,
            "location": "Paris, Île-de-France",
            "itemInSession": 0,
            "userAgent": '"Mozilla/5.0 (Macintosh; Intel Mac OS X)"',
            "method": "PUT",
            "length": 200.0,
            "song": "song_1",
            "artist": "artist_1",
            "time": CUTOFF,
            "registration": pd.Timestamp("2018-01-01"),
        }
        records = [{**defaults, **row} for row in rows]
        df = pd.DataFrame.from_records(records)[list(RAW_COLUMNS)]
        df["time"] = pd.to_datetime(df["time"])
        df["registration"] = pd.to_datetime(df["registration"])
        return df

    return _make


@pytest.fixture(scope="session")
def synthetic_events() -> pd.DataFrame:
    """Journal synthétique de taille moyenne, partagé par les tests d'intégration.

    Construit une seule fois pour toute la session : la génération coûte
    quelques secondes et le résultat est identique à chaque appel.
    """
    return make_events(n_users=60, seed=1234)
