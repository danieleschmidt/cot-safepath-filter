# CoT SafePath Filter - Makefile
# Comprehensive build, test, and deployment automation

.PHONY: help install install-dev clean test test-unit test-integration test-security test-performance lint format type-check security-check build docker-build docker-run deploy docs serve-docs release pre-commit all

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := cot-safepath-filter
PACKAGE_NAME := cot_safepath
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
DOCKER_IMAGE := $(PROJECT_NAME)
DOCKER_TAG := latest
COVERAGE_MIN := 80

# Help target
help: ## Show this help message
	@echo "CoT SafePath Filter - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make install      # Install dependencies"
	@echo "  make test         # Run all tests"
	@echo "  make lint         # Run linting"
	@echo "  make build        # Build package"
	@echo "  make all          # Run full CI pipeline"

# Installation targets
install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev,test,security,performance,integrations,docs]"
	pre-commit install
	pre-commit install --hook-type commit-msg

install-ci: ## Install CI dependencies (lightweight)
	$(PIP) install -e ".[test,security]"

# Cleaning targets
clean: ## Clean build artifacts and cache files
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/
	rm -rf $(SRC_DIR)/*.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .tox/
	rm -rf site/
	@echo "✅ Cleanup complete"

clean-docker: ## Clean Docker images and containers
	@echo "🐳 Cleaning Docker artifacts..."
	docker system prune -f
	docker image prune -f
	docker container prune -f

# Testing targets
test: ## Run all tests
	@echo "🧪 Running all tests..."
	pytest $(TEST_DIR) -v --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=html --cov-fail-under=$(COVERAGE_MIN)

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	pytest $(TEST_DIR)/unit -v --cov=$(PACKAGE_NAME) --cov-report=term-missing

test-integration: ## Run integration tests only
	@echo "🧪 Running integration tests..."
	pytest $(TEST_DIR)/integration -v --maxfail=1

test-security: ## Run security tests
	@echo "🔒 Running security tests..."
	pytest $(TEST_DIR)/security -v --tb=short

test-performance: ## Run performance tests
	@echo "⚡ Running performance tests..."
	pytest $(TEST_DIR)/performance -v --benchmark-only

test-watch: ## Run tests in watch mode
	@echo "👀 Running tests in watch mode..."
	pytest-watch $(TEST_DIR) --clear

test-coverage: ## Generate detailed coverage report
	@echo "📊 Generating coverage report..."
	pytest $(TEST_DIR) --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term-missing
	@echo "📋 Coverage report generated in htmlcov/index.html"

# Code quality targets
lint: ## Run all linting checks
	@echo "🔍 Running linting checks..."
	ruff check $(SRC_DIR) $(TEST_DIR)
	black --check $(SRC_DIR) $(TEST_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR)
	flake8 $(SRC_DIR) $(TEST_DIR)

lint-fix: ## Fix auto-fixable linting issues
	@echo "🔧 Fixing linting issues..."
	ruff check --fix $(SRC_DIR) $(TEST_DIR)
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

format: ## Format code with black and isort
	@echo "🎨 Formatting code..."
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

type-check: ## Run type checking with mypy
	@echo "🔍 Running type checks..."
	mypy $(SRC_DIR)

# Security targets
security-check: ## Run security scans
	@echo "🔒 Running security checks..."
	bandit -r $(SRC_DIR) -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true
	semgrep --config=auto $(SRC_DIR) --json --output=semgrep-report.json || true
	@echo "📋 Security reports generated: bandit-report.json, safety-report.json, semgrep-report.json"

audit: ## Run dependency audit
	@echo "🔍 Auditing dependencies..."
	pip-audit --format=json --output=audit-report.json

secrets-scan: ## Scan for secrets
	@echo "🔐 Scanning for secrets..."
	detect-secrets scan --all-files --baseline .secrets.baseline

# Build targets
build: clean ## Build distribution packages
	@echo "🏗️ Building distribution packages..."
	$(PYTHON) -m build
	@echo "✅ Build complete - packages in dist/"

build-wheel: clean ## Build wheel package only
	@echo "🏗️ Building wheel package..."
	$(PYTHON) -m build --wheel

build-sdist: clean ## Build source distribution only
	@echo "🏗️ Building source distribution..."
	$(PYTHON) -m build --sdist

# Docker targets
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker build -t $(DOCKER_IMAGE):latest .

docker-build-dev: ## Build development Docker image
	@echo "🐳 Building development Docker image..."
	docker build -f .devcontainer/Dockerfile -t $(DOCKER_IMAGE):dev .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -it --rm -p 8080:8080 $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-dev: ## Run development Docker container
	@echo "🐳 Running development Docker container..."
	docker run -it --rm -v $(PWD):/workspace -p 8080:8080 $(DOCKER_IMAGE):dev

docker-compose-up: ## Start services with docker-compose
	@echo "🐳 Starting services with docker-compose..."
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	@echo "🐳 Stopping services with docker-compose..."
	docker-compose down

# Documentation targets
docs: ## Build documentation
	@echo "📚 Building documentation..."
	mkdocs build

docs-serve: ## Serve documentation locally
	@echo "📚 Serving documentation at http://localhost:8000"
	mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "📚 Deploying documentation..."
	mkdocs gh-deploy

# Development targets
dev-setup: install-dev ## Setup development environment
	@echo "🛠️ Setting up development environment..."
	mkdir -p $(TEST_DIR)/{unit,integration,security,performance}
	mkdir -p $(DOCS_DIR)/{guides,runbooks,adr}
	mkdir -p $(SRC_DIR)/$(PACKAGE_NAME)
	mkdir -p .github/{workflows,ISSUE_TEMPLATE,PULL_REQUEST_TEMPLATE}
	@echo "✅ Development environment ready"

pre-commit: ## Run pre-commit hooks on all files
	@echo "🔧 Running pre-commit hooks..."
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	@echo "🔧 Updating pre-commit hooks..."
	pre-commit autoupdate

# Release targets
version-bump-patch: ## Bump patch version
	@echo "📦 Bumping patch version..."
	bump2version patch

version-bump-minor: ## Bump minor version
	@echo "📦 Bumping minor version..."
	bump2version minor

version-bump-major: ## Bump major version
	@echo "📦 Bumping major version..."
	bump2version major

release-prepare: ## Prepare release (build, test, lint)
	@echo "🚀 Preparing release..."
	$(MAKE) clean
	$(MAKE) install-dev
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-check
	$(MAKE) test
	$(MAKE) build
	@echo "✅ Release preparation complete"

release-publish: ## Publish to PyPI (requires authentication)
	@echo "📦 Publishing to PyPI..."
	twine check dist/*
	twine upload dist/*

release-publish-test: ## Publish to Test PyPI
	@echo "📦 Publishing to Test PyPI..."
	twine check dist/*
	twine upload --repository testpypi dist/*

# Performance targets
benchmark: ## Run benchmarks
	@echo "⚡ Running benchmarks..."
	pytest $(TEST_DIR)/performance --benchmark-only --benchmark-sort=mean

profile: ## Profile application performance
	@echo "📊 Profiling application..."
	python -m cProfile -o profile.stats -m $(PACKAGE_NAME)
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('tottime').print_stats(20)"

# Monitoring targets
health-check: ## Run application health checks
	@echo "🏥 Running health checks..."
	$(PYTHON) -c "from $(PACKAGE_NAME).health import check_health; check_health()"

metrics: ## Display application metrics
	@echo "📊 Displaying metrics..."
	$(PYTHON) -c "from $(PACKAGE_NAME).monitoring import show_metrics; show_metrics()"

# CI/CD targets
ci-install: install-ci ## CI: Install dependencies

ci-lint: lint type-check ## CI: Run linting and type checks

ci-test: test security-check ## CI: Run tests and security checks

ci-build: build ## CI: Build packages

ci-publish: release-publish ## CI: Publish to PyPI

ci-all: ci-install ci-lint ci-test ci-build ## CI: Run full pipeline

# Comprehensive targets
all: clean install-dev lint type-check security-check test build ## Run complete development pipeline
	@echo "🎉 All checks passed! Ready for commit/release."

quick-check: lint-fix type-check test-unit ## Quick development checks
	@echo "⚡ Quick checks complete"

# Database targets (if applicable)
db-init: ## Initialize database
	@echo "🗄️ Initializing database..."
	alembic upgrade head

db-migrate: ## Create database migration
	@echo "🗄️ Creating database migration..."
	alembic revision --autogenerate -m "Auto migration"

db-upgrade: ## Upgrade database to latest migration
	@echo "🗄️ Upgrading database..."
	alembic upgrade head

db-downgrade: ## Downgrade database by one migration
	@echo "🗄️ Downgrading database..."
	alembic downgrade -1

# Utility targets
install-tools: ## Install additional development tools
	@echo "🛠️ Installing additional tools..."
	$(PIP) install twine wheel bump2version

check-deps: ## Check for outdated dependencies
	@echo "📦 Checking for outdated dependencies..."
	pip list --outdated

update-deps: ## Update dependencies (be careful!)
	@echo "📦 Updating dependencies..."
	pip-upgrade

show-deps: ## Show dependency tree
	@echo "📦 Dependency tree:"
	pipdeptree

# Server targets
serve: ## Run development server
	@echo "🚀 Starting development server..."
	uvicorn $(PACKAGE_NAME).server:app --reload --host 0.0.0.0 --port 8080

serve-prod: ## Run production server
	@echo "🚀 Starting production server..."
	uvicorn $(PACKAGE_NAME).server:app --host 0.0.0.0 --port 8080 --workers 4