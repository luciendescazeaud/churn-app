# Commandes du projet.
#
# `PYTHONPATH=src` devant chaque recette rend les commandes indépendantes de
# l'état de l'installation : le code est toujours lu depuis src/, exactement
# comme le fait pytest via `pythonpath = ["src"]` dans pyproject.toml.
#
# Lance `make` sans argument pour voir la liste des commandes.

PYTHON ?= python
PYTHONPATH_SRC = PYTHONPATH=src
ARTIFACTS ?= artifacts
DEMO_ARTIFACTS = $(ARTIFACTS)/demo
EVENTS ?= data/train.parquet
USERS ?= 2000
PORT ?= 8501

IMAGE ?= churn-app

.DEFAULT_GOAL = help
.PHONY: help install test lint format train-demo train demo-csv app \
        docker-build docker-run clean

help:
	@echo "Commandes disponibles :"
	@echo "  make install     installer les dépendances et le package"
	@echo "  make test        lancer la suite de tests"
	@echo "  make lint        vérifier le style du code"
	@echo "  make format      corriger automatiquement ce qui peut l'être"
	@echo "  make train-demo  entraîner sur des données synthétiques"
	@echo "  make train       entraîner sur les données réelles ($(EVENTS))"
	@echo "  make demo-csv    produire un journal d'exemple téléchargeable"
	@echo "  make app         lancer l'application Streamlit"
	@echo "  make docker-build construire l'image Docker"
	@echo "  make docker-run   lancer l'application dans un conteneur"
	@echo "  make clean       supprimer les fichiers générés"

install:
	$(PYTHON) -m pip install -e ".[dev,app]"

test:
	$(PYTHONPATH_SRC) $(PYTHON) -m pytest

lint:
	$(PYTHONPATH_SRC) $(PYTHON) -m ruff check src tests

format:
	$(PYTHONPATH_SRC) $(PYTHON) -m ruff check --fix src tests

train-demo:
	$(PYTHONPATH_SRC) $(PYTHON) -m churn.train \
		--synthetic --users $(USERS) --out $(DEMO_ARTIFACTS)

train:
	$(PYTHONPATH_SRC) $(PYTHON) -m churn.train \
		--events $(EVENTS) --out $(ARTIFACTS)

demo-csv:
	$(PYTHONPATH_SRC) $(PYTHON) -m churn.synthetic \
		--out data/demo_events.csv --users 60

app:
	$(PYTHONPATH_SRC) $(PYTHON) -m streamlit run app/streamlit_app.py \
		--server.port $(PORT) --server.address 0.0.0.0

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8501 $(IMAGE)

clean:
	rm -rf build .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
