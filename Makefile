# ===== Makefile for common project tasks =====
# Usage: `make <target>`

.PHONY: help install install-dev lint format test test-fast clean download preprocess train-baseline train-diffusion

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ===== Setup =====
install:  ## Install dependencies
	pip install -r requirements.txt

install-dev:  ## Install with dev tools and pre-commit hooks
	pip install -r requirements.txt
	pre-commit install

# ===== Code quality =====
lint:  ## Run ruff linter
	ruff check src tests scripts

format:  ## Auto-format code with ruff
	ruff format src tests scripts
	ruff check --fix src tests scripts

# ===== Testing =====
test:  ## Run all tests
	pytest tests/ -v

test-fast:  ## Run only fast tests (skip slow/gpu)
	pytest tests/ -v -m "not slow and not gpu"

# ===== Cleaning =====
clean:  ## Remove cache and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# ===== Data pipeline =====
download:  ## Download osu!mania 4K data (requires OSU_API_KEY)
	python scripts/download_data.py --mode mania-4k --status ranked --output data/raw

preprocess:  ## Preprocess raw data into training-ready format
	python scripts/preprocess_data.py --input data/raw --output data/processed

# ===== Training =====
train-baseline:  ## Train autoregressive Transformer baseline
	python scripts/train.py model=transformer_baseline data=osu_mania_4k

train-diffusion:  ## Train discrete diffusion model
	python scripts/train.py model=diffusion data=osu_mania_4k
