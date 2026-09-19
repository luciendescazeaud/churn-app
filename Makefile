PYTHON ?= python
PYTHONPATH_SRC = PYTHONPATH=src
ARTIFACTS ?= artifacts
DEMO_ARTIFACTS = $(ARTIFACTS)/demo
EVENTS ?= data/train.parquet
USERS ?= 2000
PORT ?= 8501

.DEFAULT_GOAL = help
.PHONY: help install test lint format train-demo train demo-csv app clean

help:
	@echo "Commandes disponibles :"
	@echo "  make install     installer les dependances et le package"
	@echo "  make test        lancer la suite de tests"
	@echo "  make lint        verifier le style du code"
	@echo "  make format      corriger automatiquement ce qui peut l'etre"
	@echo "  make train-demo  entrainer sur des donnees synthetiques"
	@echo "  make train       entrainer sur les donnees reelles ($(EVENTS))"
	@echo "  make demo-csv    produire un journal d'exemple telechargeable"
	@echo "  make app         lancer l'application Streamlit"
	@echo "  make clean       supprimer les fichiers generes"

install:
	$(PYTHON) -m pip install -e ".[dev,app]"

test:
	$(PYTHONPATH_SRC) $(PYTHON) -m pytest

lint:
	$(PYTHONPATH_SRC) $(PYTHON) -m ruff check src tests

format:
	$(PYTHONPATH_SRC) $(PYTHON) -m ruff check --fix src tests

train-demo:
	$(PYTHONPATH_SRC) $(PYTHON) -m churn.train --synthetic --users $(USERS) --out $(DEMO_ARTIFACTS)

train:
	$(PYTHONPATH_SRC) $(PYTHON) -m churn.train --events $(EVENTS) --out $(ARTIFACTS)

demo-csv:
	$(PYTHONPATH_SRC) $(PYTHON) -m churn.synthetic --out data/demo_events.csv --users 60

app:
	$(PYTHONPATH_SRC) $(PYTHON) -m streamlit run app/streamlit_app.py --server.port $(PORT) --server.address 0.0.0.0

clean:
	rm -rf build .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
