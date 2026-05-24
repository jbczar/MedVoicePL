# ============================================================================
# MedVoicePL — Makefile
# ============================================================================

.DEFAULT_GOAL := help

# ------------------------------------------
# Environment
# ------------------------------------------
.PHONY: install
install: ## Install all workspace dependencies via uv (+ Presidio anonymization extra)
	uv sync --all-packages --extra anonymization

.PHONY: install-frontend
install-frontend: ## Install frontend (Next.js) dependencies
	cd application/frontend && npm install

# ------------------------------------------
# Code Quality
# ------------------------------------------
.PHONY: lint
lint: ## Run ruff linter across the monorepo
	uv run ruff check .

.PHONY: lint-fix
lint-fix: ## Run ruff linter with auto-fix
	uv run ruff check --fix .

.PHONY: format
format: ## Format code with ruff
	uv run ruff format .

.PHONY: format-check
format-check: ## Check code formatting without modifying files
	uv run ruff format --check .

.PHONY: typecheck
typecheck: ## Run mypy type checking
	uv run mypy dataset_pipeline/src application/on_premise/src application/cloud/src application/gateway/src application/shared/src

.PHONY: check
check: lint format-check typecheck ## Run all code quality checks

# ------------------------------------------
# Testing
# ------------------------------------------
.PHONY: test
test: ## Run all unit tests
	uv run pytest

.PHONY: test-on-premise
test-on-premise: ## Run on-premise module tests
	uv run pytest application/on_premise/tests/

.PHONY: test-cloud
test-cloud: ## Run cloud module tests
	uv run pytest application/cloud/tests/

.PHONY: test-gateway
test-gateway: ## Run gateway module tests
	uv run pytest application/gateway/tests/

.PHONY: test-shared
test-shared: ## Run shared module tests
	uv run pytest application/shared/tests/

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	uv run pytest --cov=. --cov-report=html

# ------------------------------------------
# Docker
# ------------------------------------------
.PHONY: up
up: ## Start all services via Docker Compose
	docker compose up --build -d

.PHONY: down
down: ## Stop all services
	docker compose down

.PHONY: logs
logs: ## Tail logs from all services
	docker compose logs -f

.PHONY: restart
restart: down up ## Restart all services

# ------------------------------------------
# Gateway (Phase 3b) — native uvicorn, no Docker
# ------------------------------------------
GATEWAY_PID := /tmp/medvoicepl-gateway.pid
GATEWAY_LOG := /tmp/medvoicepl-gateway.log

.PHONY: gateway-up
gateway-up: ## Start gateway in DEMO mode (background, native uvicorn)
	@if [ -f $(GATEWAY_PID) ] && kill -0 $$(cat $(GATEWAY_PID)) 2>/dev/null; then \
		echo "gateway already running (pid $$(cat $(GATEWAY_PID)))"; \
	else \
		GATEWAY_DEMO_MODE=true GATEWAY_HOST=127.0.0.1 GATEWAY_PORT=8080 \
		nohup uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8080 \
			> $(GATEWAY_LOG) 2>&1 & echo $$! > $(GATEWAY_PID); \
		echo "gateway started (pid $$(cat $(GATEWAY_PID))), log: $(GATEWAY_LOG)"; \
	fi

.PHONY: gateway-down
gateway-down: ## Stop gateway started via gateway-up
	@if [ -f $(GATEWAY_PID) ]; then \
		PID=$$(cat $(GATEWAY_PID)); \
		if kill -0 $$PID 2>/dev/null; then kill $$PID && echo "gateway stopped (pid $$PID)"; \
		else echo "gateway pid $$PID not running"; fi; \
		rm -f $(GATEWAY_PID); \
	else echo "no gateway pid file"; fi

.PHONY: gateway-logs
gateway-logs: ## Tail the gateway log
	@tail -f $(GATEWAY_LOG)

.PHONY: gateway-status
gateway-status: ## Check gateway health endpoint
	@curl -sf http://127.0.0.1:8080/v1/health | python3 -m json.tool || echo "gateway not reachable"

.PHONY: smoke-3b
smoke-3b: ## End-to-end smoke against running gateway (login → finalize → logout)
	uv run python application/gateway/scripts/smoke_e2e.py

# ------------------------------------------
# Live mode (real on-prem + cloud + gateway, native uvicorn)
# ------------------------------------------
ONPREM_PID := /tmp/medvoicepl-onprem.pid
ONPREM_LOG := /tmp/medvoicepl-onprem.log
CLOUD_PID := /tmp/medvoicepl-cloud.pid
CLOUD_LOG := /tmp/medvoicepl-cloud.log

.PHONY: preflight-live
preflight-live: ## Verify env, weights, NFZ, audio for live mode
	uv run python application/gateway/scripts/preflight_live.py

.PHONY: onprem-up
onprem-up: ## Start on-prem service on 127.0.0.1:8001 (real MLX/pyannote/Bielik)
	@if [ -f $(ONPREM_PID) ] && kill -0 $$(cat $(ONPREM_PID)) 2>/dev/null; then \
		echo "on-prem already running (pid $$(cat $(ONPREM_PID)))"; \
	else \
		nohup uv run uvicorn on_premise.api.app:app --host 127.0.0.1 --port 8001 \
			> $(ONPREM_LOG) 2>&1 & echo $$! > $(ONPREM_PID); \
		echo "on-prem started (pid $$(cat $(ONPREM_PID))), log: $(ONPREM_LOG)"; \
	fi

.PHONY: onprem-down
onprem-down: ## Stop on-prem service
	@if [ -f $(ONPREM_PID) ]; then \
		PID=$$(cat $(ONPREM_PID)); \
		if kill -0 $$PID 2>/dev/null; then kill $$PID && echo "on-prem stopped (pid $$PID)"; \
		else echo "on-prem pid $$PID not running"; fi; \
		rm -f $(ONPREM_PID); \
	else echo "no on-prem pid file"; fi

.PHONY: cloud-up
cloud-up: ## Start cloud service on 127.0.0.1:8002 (real Vertex/Qdrant/embedder)
	@if [ -f $(CLOUD_PID) ] && kill -0 $$(cat $(CLOUD_PID)) 2>/dev/null; then \
		echo "cloud already running (pid $$(cat $(CLOUD_PID)))"; \
	else \
		nohup uv run uvicorn cloud.api.app:app --host 127.0.0.1 --port 8002 \
			> $(CLOUD_LOG) 2>&1 & echo $$! > $(CLOUD_PID); \
		echo "cloud started (pid $$(cat $(CLOUD_PID))), log: $(CLOUD_LOG)"; \
	fi

.PHONY: cloud-down
cloud-down: ## Stop cloud service
	@if [ -f $(CLOUD_PID) ]; then \
		PID=$$(cat $(CLOUD_PID)); \
		if kill -0 $$PID 2>/dev/null; then kill $$PID && echo "cloud stopped (pid $$PID)"; \
		else echo "cloud pid $$PID not running"; fi; \
		rm -f $(CLOUD_PID); \
	else echo "no cloud pid file"; fi

.PHONY: live-up
live-up: onprem-up cloud-up ## Start on-prem + cloud (gateway separately, in live mode)
	@echo "Tail logs: tail -f $(ONPREM_LOG) $(CLOUD_LOG)"
	@echo "Wait ~30-60 s for cold-start (model loads). Then: make gateway-up-live"

.PHONY: live-down
live-down: cloud-down onprem-down gateway-down ## Stop all live services

.PHONY: gateway-up-live
gateway-up-live: ## Start gateway in LIVE mode (talks to real on-prem + cloud)
	@if [ -f $(GATEWAY_PID) ] && kill -0 $$(cat $(GATEWAY_PID)) 2>/dev/null; then \
		echo "gateway already running (pid $$(cat $(GATEWAY_PID)))"; \
	else \
		GATEWAY_DEMO_MODE=false GATEWAY_HOST=127.0.0.1 GATEWAY_PORT=8080 \
		nohup uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8080 \
			> $(GATEWAY_LOG) 2>&1 & echo $$! > $(GATEWAY_PID); \
		echo "gateway started in LIVE mode (pid $$(cat $(GATEWAY_PID)))"; \
	fi

# ------------------------------------------
# Frontend (Phase 3c) — Next.js dev/build/start
# ------------------------------------------
FRONTEND_DIR := application/frontend
FRONTEND_PID := /tmp/medvoicepl-frontend.pid
FRONTEND_LOG := /tmp/medvoicepl-frontend.log

.PHONY: frontend-install
frontend-install: ## Install frontend dependencies (npm install in application/frontend)
	cd $(FRONTEND_DIR) && npm install --no-audit --no-fund

.PHONY: frontend-dev
frontend-dev: ## Run Next.js dev server (foreground, hot reload)
	cd $(FRONTEND_DIR) && npm run dev

.PHONY: frontend-build
frontend-build: ## Build the production frontend bundle
	cd $(FRONTEND_DIR) && npm run build

.PHONY: frontend-up
frontend-up: ## Start built frontend on :3000 in background
	@PORT_PID=$$(lsof -ti :3000 2>/dev/null); \
	if [ -n "$$PORT_PID" ]; then \
		echo "port 3000 already in use by pid $$PORT_PID — run 'make frontend-down' first"; \
		exit 1; \
	fi; \
	cd $(FRONTEND_DIR) && nohup npm start > $(FRONTEND_LOG) 2>&1 & \
	sleep 2; \
	REAL_PID=$$(lsof -ti :3000 2>/dev/null); \
	if [ -n "$$REAL_PID" ]; then \
		echo $$REAL_PID > $(FRONTEND_PID); \
		echo "frontend started (pid $$REAL_PID on :3000), log: $(FRONTEND_LOG)"; \
	else \
		echo "frontend failed to bind :3000 — check $(FRONTEND_LOG)"; \
		exit 1; \
	fi

.PHONY: frontend-down
frontend-down: ## Stop frontend on :3000 (port-aware, not pidfile-dependent)
	@PORT_PID=$$(lsof -ti :3000 2>/dev/null); \
	if [ -n "$$PORT_PID" ]; then \
		kill $$PORT_PID && echo "frontend stopped (pid $$PORT_PID on :3000)"; \
	else \
		echo "no process on :3000"; \
	fi; \
	rm -f $(FRONTEND_PID)

.PHONY: frontend-test
frontend-test: ## Run vitest suite for the frontend
	cd $(FRONTEND_DIR) && npm test

.PHONY: demo-up
demo-up: gateway-up frontend-up ## Boot demo stack (gateway in demo mode + frontend on :3000)
	@echo "Demo ready: open http://localhost:3000 (login demo_doctor / demo_password)"

.PHONY: demo-down
demo-down: frontend-down gateway-down ## Stop demo stack

.PHONY: smoke-3b-live
smoke-3b-live: ## E2E smoke with real audio (Graves-Basedow → Tiamazol+Propranolol)
	uv run python application/gateway/scripts/smoke_e2e.py \
		--audio-file dataset_pipeline/output/audio/endo-gra/dialogue.wav \
		--expect-drug Tiamazol --expect-drug Propranolol \
		--timeout 600

# ------------------------------------------
# Dataset Pipeline
# ------------------------------------------
.PHONY: generate-dataset
generate-dataset: ## Run the synthetic dataset generation pipeline
	uv run python dataset_pipeline/scripts/run_generation.py

.PHONY: validate-dataset
validate-dataset: ## Validate the generated dataset integrity
	uv run python dataset_pipeline/scripts/validate_dataset.py

# ------------------------------------------
# Evaluation
# ------------------------------------------
.PHONY: eval
eval: ## Run all evaluation benchmarks
	uv run python research_and_testing/scripts/run_all_benchmarks.py

.PHONY: eval-report
eval-report: ## Generate evaluation report (HTML/PDF)
	uv run python research_and_testing/scripts/generate_eval_report.py

# ------------------------------------------
# Models
# ------------------------------------------
.PHONY: download-models
download-models: ## Download required ML models (Whisper, SLM)
	bash application/on_premise/models/download_models.sh

# ------------------------------------------
# Cleanup
# ------------------------------------------
.PHONY: clean
clean: ## Remove build artifacts, caches, and generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage

# ------------------------------------------
# Help
# ------------------------------------------
.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
