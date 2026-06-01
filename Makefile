.PHONY: help build up down logs download index reindex shell test lint fresh

COMPOSE = docker compose
API_SERVICE = api
PYTHON = $(shell [ -d .venv ] && echo .venv/bin/python || echo python3)

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Image & containers ────────────────────────────────────────────────────────

build: ## Build (or rebuild) the ego-api Docker image
	$(COMPOSE) build --pull $(API_SERVICE)

up: ## Start API in the background
	$(COMPOSE) up -d

down: ## Stop and remove all containers (data volumes are preserved)
	$(COMPOSE) down

logs: ## Tail live logs from the API container
	$(COMPOSE) logs -f $(API_SERVICE)

restart: ## Restart only the API container (e.g. after a code change)
	$(COMPOSE) restart $(API_SERVICE)

# ── Data pipeline ─────────────────────────────────────────────────────────────

download: ## Download data from Kaggle into data/ (reads KAGGLE_DATASET from .env)
	$(PYTHON) scripts/ingest.py

index: ## Full pipeline: download from Kaggle → convert → build Turbovec index
	$(COMPOSE) --profile tools run --rm indexer

reindex: ## Force re-download + re-index (ignores all locally cached files)
	$(COMPOSE) --profile tools run --rm indexer --rebuild

# ── Development ───────────────────────────────────────────────────────────────

shell: ## Open an interactive shell inside the running API container
	$(COMPOSE) exec $(API_SERVICE) /bin/bash

test: ## Run the test suite inside a fresh container
	$(COMPOSE) run --rm --no-deps $(API_SERVICE) pytest -v tests/

lint: ## Lint the source tree with ruff
	$(COMPOSE) run --rm --no-deps $(API_SERVICE) ruff check .

# ── Shortcuts ─────────────────────────────────────────────────────────────────

fresh: down build up ## Tear down, rebuild image, and bring everything back up
