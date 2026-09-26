# ===== Makefile for common project tasks =====
# Usage: `make <target>`

.PHONY: help install install-dev lint format test test-fast clean download validate-tokenizer manifest preprocess overfit train-baseline train-diffusion

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

validate-tokenizer:  ## Check tokenizer invariants on every chart in data/raw
	python scripts/validate_tokenizer.py --root data/raw

manifest:  ## One row per chart with split and SR -> data/manifest.csv
	python scripts/build_manifest.py

preprocess:  ## Token cache for every chart in the manifest -> data/cache/tokens
	python scripts/preprocess_data.py

overfit:  ## Milestone check: overfit 10 charts, then rebuild one from all-MASK
	python scripts/train.py --overfit 10

# ===== Training =====
train-baseline:  ## AR baseline (Yi et al. setting): not implemented yet (W5)
	@echo "AR-4 / AR-32 baseline is not implemented yet (design doc, W5)"; exit 1

train-diffusion:  ## Train the D-32 denoiser on the train split
	python scripts/train.py --steps 50000
