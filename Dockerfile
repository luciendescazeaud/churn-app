# Image de l'application de prédiction du churn.
#
# Python 3.12 : même version qu'en local et en intégration continue. C'est la
# condition de base de la reproductibilité — un modèle sérialisé dépend des
# versions de librairies qui l'ont produit.
#
# Construction et lancement :
#     docker build -t churn-app .
#     docker run --rm -p 8501:8501 churn-app

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dépendances système. LightGBM parallélise l'entraînement via OpenMP : sa
# partie compilée réclame libgomp, absente des images « slim ». Sans elle,
# `import lightgbm` échoue sur un fichier .so introuvable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Les dépendances d'abord, le code applicatif ensuite. Docker met chaque
# instruction en cache : tant que pyproject.toml et src/ ne changent pas,
# cette couche — la plus longue — n'est pas reconstruite.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Installation normale, pas modifiable : le package est copié dans
# site-packages comme n'importe quelle librairie. L'extra `app` ajoute
# Streamlit ; les outils de développement restent dehors.
RUN pip install ".[app]"

COPY app/ ./app/

# Artefacts produits pendant la construction, à partir de données
# synthétiques. L'image est ainsi autonome et ne contient aucune donnée de
# compétition. Pour servir le modèle réel, monter un dossier au lancement :
#     docker run -v "$(pwd)/artifacts:/app/artifacts" -p 8501:8501 churn-app
RUN python -m churn.train --synthetic --users 2000 --out artifacts/demo

# Utilisateur non privilégié : un processus compromis ne doit pas être root.
RUN useradd --create-home --uid 1000 churn && chown -R churn:churn /app
USER churn

EXPOSE 8501

# Streamlit expose un point de santé ; Docker s'en sert pour savoir si le
# conteneur répond vraiment, et pas seulement s'il tourne encore.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# --server.address=0.0.0.0 est indispensable : à l'intérieur du conteneur,
# localhost ne désigne que le conteneur lui-même, et rien ne sortirait.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
