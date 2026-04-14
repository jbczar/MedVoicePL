# ============================================================================
# MedVoicePL — Makefile
# ============================================================================

.DEFAULT_GOAL := help

# ------------------------------------------
# Environment
# ------------------------------------------
.PHONY: install
install: ## Install all workspace dependencies via uv
	uv sync --all-packages

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
