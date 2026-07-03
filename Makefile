.PHONY: help build up down logs download index reindex shell test lint fresh \
        acr-login acr-push deploy-ecs deploy-ack ack-secrets proof

COMPOSE     = docker compose
API_SERVICE = api
PYTHON      = $(shell [ -d .venv ] && echo .venv/bin/python || echo python3)

# ── Alibaba Cloud settings (override on CLI or via .env) ─────────────────────
# ACR Enterprise instance endpoint (Singapore)
ACR_REGISTRY   ?= dreamer-registry.ap-southeast-1.cr.aliyuncs.com
ACR_NAMESPACE  ?= ego
API_IMAGE       = $(ACR_REGISTRY)/$(ACR_NAMESPACE)/ego-api:latest
FRONTEND_IMAGE  = $(ACR_REGISTRY)/$(ACR_NAMESPACE)/ego-frontend:latest
ECS_IP         ?=   # set on CLI: make deploy-ecs ECS_IP=47.xx.xx.xx

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Image & containers ────────────────────────────────────────────────────────

build: ## Build (or rebuild) all Docker images locally
	$(COMPOSE) build --pull

up: ## Start all services in the background
	$(COMPOSE) up -d

down: ## Stop and remove all containers (data volumes are preserved)
	$(COMPOSE) down

logs: ## Tail live logs from the API container
	$(COMPOSE) logs -f $(API_SERVICE)

restart: ## Restart only the API container (e.g. after a code change)
	$(COMPOSE) restart $(API_SERVICE)

# ── Alibaba Cloud Container Registry ─────────────────────────────────────────

acr-login: ## Log in to Alibaba Cloud Container Registry (ACR Enterprise)
	docker login --username=$(ACR_USER) $(ACR_REGISTRY)

acr-create-repos: ## Create ACR namespace + repositories (required before first push on Enterprise)
	@echo "Creating namespace '$(ACR_NAMESPACE)' on $(ACR_REGISTRY)…"
	aliyun cr CreateNamespace --InstanceId $(ACR_INSTANCE_ID) \
	    --NamespaceName $(ACR_NAMESPACE) --AutoCreateRepo true \
	    --DefaultRepoType PUBLIC 2>/dev/null || true
	@echo "Creating repository ego-api…"
	aliyun cr CreateRepository --InstanceId $(ACR_INSTANCE_ID) \
	    --NamespaceName $(ACR_NAMESPACE) --RepoName ego-api \
	    --RepoType PRIVATE --Summary "Ego API" 2>/dev/null || true
	@echo "Creating repository ego-frontend…"
	aliyun cr CreateRepository --InstanceId $(ACR_INSTANCE_ID) \
	    --NamespaceName $(ACR_NAMESPACE) --RepoName ego-frontend \
	    --RepoType PRIVATE --Summary "Ego Frontend" 2>/dev/null || true
	@echo "✓ Repositories ready. Now run: make acr-push"

acr-push:  ## Build images and push to ACR
	docker tag ego-api:latest $(API_IMAGE)
	docker tag ego-frontend:latest $(FRONTEND_IMAGE)
	docker push $(API_IMAGE)
	docker push $(FRONTEND_IMAGE)
	@echo "✓ Pushed to $(ACR_REGISTRY)/$(ACR_NAMESPACE)"


# ── Alibaba Cloud deployment ──────────────────────────────────────────────────

deploy-ecs: ## Deploy to Alibaba Cloud ECS via Docker Compose (set ECS_IP=...)
	@[ -n "$(ECS_IP)" ] || (echo "Usage: make deploy-ecs ECS_IP=<public-ip>" && exit 1)
	bash scripts/deploy_acs.sh --mode ecs --ecs-ip $(ECS_IP) \
	    --region $(ACR_REGION) --namespace $(ACR_NAMESPACE)

deploy-ack: ## Deploy / update to Alibaba Cloud ACK (kubectl must be configured)
	bash scripts/deploy_acs.sh --mode ack \
	    --region $(ACR_REGION) --namespace $(ACR_NAMESPACE)

ack-secrets: ## Create/update the ego-secrets Secret on ACK from .env
	@. ./.env && kubectl create secret generic ego-secrets \
	    --namespace=ego \
	    --from-literal=DASHSCOPE_API_KEY="$$DASHSCOPE_API_KEY" \
	    --from-literal=QWEN_MODEL="$${QWEN_MODEL:-qwen-plus}" \
	    --dry-run=client -o yaml | kubectl apply -f -
	@echo "✓ ego-secrets applied"

ack-acr-secret: ## Create ACR pull secret on ACK (set ACR_USER and ACR_PASS)
	@[ -n "$(ACR_USER)" ] || (echo "Usage: make ack-acr-secret ACR_USER=<user> ACR_PASS=<pwd>" && exit 1)
	kubectl create secret docker-registry acr-secret \
	    --namespace=ego \
	    --docker-server=$(ACR_REGISTRY) \
	    --docker-username=$(ACR_USER) \
	    --docker-password=$(ACR_PASS) \
	    --dry-run=client -o yaml | kubectl apply -f -

proof: ## Run the Alibaba Cloud proof-of-deployment script
	$(PYTHON) alibaba_cloud_proof.py

# ── Data pipeline ─────────────────────────────────────────────────────────────

download: ## Download data into data/ (reads DATASET_BASE_URL from .env)
	$(PYTHON) scripts/ingest.py

index: ## Full pipeline: download → convert → build Turbovec index
	$(COMPOSE) --profile tools run --rm indexer

reindex: ## Force re-download + re-index (ignores all cached files)
	$(COMPOSE) --profile tools run --rm indexer --rebuild

# ── Development ───────────────────────────────────────────────────────────────

shell: ## Open an interactive shell inside the running API container
	$(COMPOSE) exec $(API_SERVICE) /bin/bash

test: ## Run the test suite inside a fresh container
	$(COMPOSE) run --rm --no-deps $(API_SERVICE) pytest -v tests/

lint: ## Lint the source tree with ruff
	$(COMPOSE) run --rm --no-deps $(API_SERVICE) ruff check .

# ── Shortcuts ─────────────────────────────────────────────────────────────────

fresh: down build up ## Tear down, rebuild images, and bring everything back up
