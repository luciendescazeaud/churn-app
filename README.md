# Prédiction de churn — pipeline, application et conteneur

[![CI](https://github.com/luciendescazeaud/churn-app/actions/workflows/ci.yml/badge.svg)](https://github.com/luciendescazeaud/churn-app/actions/workflows/ci.yml)

Prédire la résiliation d'abonnés d'un service de streaming musical à partir de
leurs journaux d'événements, et servir ce modèle par une application web
conteneurisée.

Le projet couvre la chaîne complète : journaux bruts → nettoyage → feature
engineering → modèle → artefacts versionnés → application Streamlit → image
Docker, avec tests automatisés et intégration continue.

---

## Origine du travail

Le modèle et le feature engineering proviennent d'un projet Kaggle réalisé en
binôme avec Martin dans le cadre du cours *Python for Data Science*.

Tout ce qui constitue le présent rendu est un travail individuel :
restructuration du code en package Python, correction de trois défauts du
pipeline d'origine, révision du protocole de choix du seuil, générateur de
données synthétiques, suite de tests, application Streamlit, Dockerfile,
intégration continue et documentation.

---

## Démarrage rapide

### Avec Docker

```bash
docker build -t churn-app .
docker run --rm -p 8501:8501 churn-app
```

Puis `http://localhost:8501`.

L'image est autonome : elle entraîne un modèle sur des données synthétiques
pendant sa construction et ne contient aucune donnée de compétition.

### En local

```bash
python3.12 -m venv .venv
source .venv/bin/activate

make install      # installe le package et ses dépendances
make test         # 94 tests
make train-demo   # produit artifacts/demo/
make app          # ouvre l'application
```

`make` sans argument liste toutes les commandes disponibles.

---

## L'application

Trois pages.

**Performance du modèle** — balanced accuracy, AUC, rappels par classe,
matrice de confusion, courbe de balanced accuracy en fonction du seuil,
importance des features, et la provenance complète du modèle chargé.

**Prédire sur un journal** — un fichier de logs importé, ou un jeu d'exemple
synthétique généré à la volée. Le fichier suit exactement le chemin de
l'entraînement : validation du schéma, nettoyage, feature engineering,
alignement des colonnes, puis modèle. Le résultat est le classement des
utilisateurs par probabilité de résiliation, téléchargeable en CSV.

**À propos** — le protocole méthodologique.

Il n'y a délibérément pas de formulaire de saisie manuelle des features : le
modèle travaille sur des variables issues de l'agrégation de journaux, et
faire mine qu'un utilisateur peut les saisir donnerait une démonstration
trompeuse.

---

## Méthodologie

### Protocole anti-fuite

Les features d'un utilisateur ne sont construites qu'à partir de ses
événements **antérieurs à l'instant de prédiction**, et la cible est observée
dans la fenêtre de dix jours qui suit :

```
────────────── historique observable ──────────────┤ cutoff ├── fenêtre ──┤
              features construites ici                         cible ici
```

Le cutoff d'entraînement est reculé d'exactement une fenêtre de churn par
rapport à celui de l'inférence, de sorte que les deux observent le même
horizon. Cet invariant est encodé dans `SplitConfig` et vérifié par un test.

Trois garde-fous se superposent :

1. le journal est tronqué au cutoff avant tout calcul, dans `build_features` ;
2. les pages de résiliation et de rétrogradation sont exclues des comptages
   de pages et du calcul d'entropie ;
3. un contrôle final rejette toute feature dont le nom contient un jeton
   interdit (`cancel`, `downgrade`…).

Le test le plus important de la suite vérifie cette propriété de bout en bout :
construire les features, puis les reconstruire après avoir ajouté des
événements postérieurs au cutoff — y compris une résiliation — doit donner un
résultat **rigoureusement identique**.

### Choix du seuil

La cible est rare : le seuil par défaut de 0,5 ne signalerait presque
personne. Le seuil est donc un paramètre appris, choisi pour maximiser la
balanced accuracy, et **sauvegardé avec le modèle**.

Il est calibré sur des prédictions **hors échantillon**, obtenues par
validation croisée stratifiée en cinq plis : chaque utilisateur est prédit par
un modèle qui ne l'a pas vu à l'entraînement. Le calibrer sur le jeu ayant
servi à l'arrêt anticipé donnerait une balanced accuracy optimiste sur deux
plans à la fois.

### Modèle

LightGBM, avec arrêt anticipé sur l'AUC dans chaque pli. Le modèle final est
réentraîné sur l'ensemble des données avec le nombre d'arbres moyen des plis.

---

## Structure

```
.
├── app/
│   └── streamlit_app.py       interface — aucune logique métier
├── src/churn/
│   ├── config.py              schéma des données, protocole temporel
│   ├── data.py                import, validation, filtrage, cible
│   ├── features.py            13 blocs de features, alignement des colonnes
│   ├── synthetic.py           générateur de journaux
│   ├── evaluation.py          seuil et métriques
│   ├── train.py               entraînement → artefacts
│   └── predict.py             artefacts → prédictions
├── tests/                     94 tests
├── .github/workflows/ci.yml   style, tests, image, publication
├── Dockerfile
├── Makefile                   toutes les commandes du projet
└── pyproject.toml             package et dépendances
```

Le sens des dépendances est strict : `predict.py` n'importe pas `train.py`.
L'application n'a donc jamais besoin du code d'entraînement.

---

## Les artefacts

L'entraînement produit cinq fichiers, et ce sont les seuls que l'application
consomme :

| Fichier | Contenu |
|---|---|
| `model.joblib` | le modèle entraîné |
| `feature_columns.json` | l'ordre exact des colonnes attendues |
| `metrics.json` | seuil, métriques et provenance |
| `feature_importance.csv` | contribution de chaque feature |
| `threshold_scan.csv` | balanced accuracy en fonction du seuil |

`metrics.json` porte la provenance complète : commit Git, graine, versions de
Python et des librairies, cutoffs, et une empreinte du jeu d'entraînement.
Elle permet de vérifier des mois plus tard qu'un modèle sauvegardé a bien été
entraîné sur les données qu'on croit.

---

## Tests

```bash
make test
```

94 tests, sans aucune donnée externe : les fixtures sont soit des journaux de
trois lignes écrits à la main, soit des données synthétiques reproductibles.
La suite tourne donc partout, y compris en intégration continue.

| Fichier | Couvre |
|---|---|
| `test_data.py` | import, validation de schéma, filtrage, construction de la cible |
| `test_features.py` | agrégats, absence de fuite, alignement des colonnes |
| `test_synthetic.py` | schéma, déterminisme, propriétés du générateur |
| `test_training.py` | seuil, métriques, artefacts, prédiction |
| `test_app.py` | démarrage de l'application et parcours de prédiction |

---

## Intégration continue

À chaque poussée et chaque pull request :

1. **style et tests** — `ruff` puis `pytest` avec rapport de couverture ;
2. **construction de l'image** — puis lancement du conteneur et interrogation
   du point de santé de Streamlit, car une image peut se construire sans
   jamais démarrer ;
3. **publication** sur Docker Hub, uniquement depuis `main` et seulement si
   les identifiants sont configurés.

Pour activer la publication, définir dans les réglages du dépôt la variable
`DOCKERHUB_USERNAME` et le secret `DOCKERHUB_TOKEN`. En leur absence, l'étage
est ignoré sans faire échouer la CI.

---

## Reproductibilité

- **Même version de Python partout** — 3.12 en local, en CI et dans l'image.
- **Dépendances déclarées** dans `pyproject.toml`, avec des planchers de
  version explicites.
- **Graines fixées** pour le générateur de données, la validation croisée et
  le modèle.
- **Provenance enregistrée** dans `metrics.json` à chaque entraînement.
- **Commandes uniques** — la CI exécute les cibles du `Makefile`, donc
  exactement ce qui tourne en local.
- **Aucune donnée versionnée** : `data/` et `artifacts/` sont ignorés par Git.

---

## Les données

Les journaux d'origine proviennent d'une compétition Kaggle et **ne sont pas
redistribués** : ni dans le dépôt, ni dans l'image. Pour entraîner sur les
données réelles, les placer dans `data/train.parquet` puis :

```bash
make train
```

Pour servir ce modèle dans le conteneur plutôt que celui construit sur données
synthétiques :

```bash
docker run --rm -p 8501:8501 -v "$(pwd)/artifacts:/app/artifacts" churn-app
```

Le générateur synthétique reproduit le schéma des 19 colonnes d'origine, la
mécanique du churn, les sessions déconnectées au `userId` vide, et un
désengagement progressif des futurs résiliants.

---

## Limites

- Les métriques affichées par défaut dans l'application sont celles d'un
  modèle entraîné sur **données synthétiques**. Elles valident la chaîne, pas
  la performance sur données réelles.
- L'importance des features mesure la fréquence d'utilisation dans les arbres,
  ce qui n'est pas une mesure de causalité.
- Le seuil maximise la balanced accuracy. Un objectif commercial différent —
  un budget de rétention contraint, par exemple — appellerait un autre point
  de fonctionnement, que `threshold_scan.csv` permet de choisir.

---

## Image publiée

L'image est publiée sur Docker Hub à chaque poussée sur `main`. Pour lancer
l'application sans cloner ce dépôt :

```bash
docker run --rm -p 8501:8501 luciendescaz/churn-app:latest
```

Deux étiquettes sont publiées. `latest` suit la dernière version de `main`.
Une seconde étiquette porte l'identifiant du commit qui a produit l'image, et
reste immuable — c'est elle qu'il faut utiliser pour reproduire un résultat
précis.

`https://hub.docker.com/r/luciendescaz/churn-app`

---

## Licence

MIT.
