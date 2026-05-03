.DEFAULT_GOAL := help

POETRY := poetry run
PYTHON := $(POETRY) python

.PHONY: help install ci \
	lint lint-fix format \
	test test-cov \
	data_download \
	data_combine_flights data_merge_weather data_merge_sources data_clean \
	data_process \
	validate_flights validate_weather validate_merged validate_all validate_serve \
	features features_granular eda \
	models models_granular models_weights models_catboost models_ensemble \
	mlflow_ui \
	all pipeline

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "Flight Delay Predictor — Make targets (run from repo root)"
	@echo ""
	@echo "Setup and code quality:"
	@echo "  make install       Poetry install"
	@echo "  make lint          Ruff check"
	@echo "  make lint-fix      Ruff check --fix"
	@echo "  make format        Ruff format"
	@echo "  make test          pytest tests/"
	@echo "  make test-cov      pytest with coverage (dev: pytest-cov)"
	@echo "  make ci            lint then test"
	@echo ""
	@echo "1 — Data acquisition (airlines, airports, flights, weather):"
	@echo "  make data_download"
	@echo ""
	@echo "2 — Interim processing:"
	@echo "  make data_combine_flights"
	@echo "  make data_merge_weather"
	@echo ""
	@echo "3 — Interim validation (scripts; notebooks are .ipynb beside src/notebooks/eda.py):"
	@echo "  make validate_flights     [data/interim/flights_combined.parquet]"
	@echo "  make validate_weather     [data/interim/weather/weather_combined.csv]"
	@echo ""
	@echo "4 — Final merge, clean, merged validation:"
	@echo "  make data_merge_sources"
	@echo "  make data_clean"
	@echo "  make validate_merged      [.env PROJECT_ROOT; reports/validation_results.json]"
	@echo "  make validate_serve       [http.server :8080 for HTML dashboard]"
	@echo ""
	@echo "5 — Features:"
	@echo "  make features             [FeaturePipeline]"
	@echo "  make features_granular    [three separate scripts]"
	@echo ""
	@echo "6 — EDA:"
	@echo "  make eda                  [src/notebooks/eda.py -> reports/figures/]"
	@echo "  Jupyter: src/notebooks/*.ipynb, src/data/validate_*_metadata.ipynb"
	@echo ""
	@echo "7 — Models (MLflow inside training/compare scripts):"
	@echo "  make models               [ModelPipeline]"
	@echo "  make models_granular      [four scripts: weights, train, compare, tune]"
	@echo "  make models_weights       [optional LR weight comparison]"
	@echo "  make models_catboost      [optional; needs catboost]"
	@echo "  make models_ensemble      [optional soft voting; needs pkls]"
	@echo "  make mlflow_ui"
	@echo ""
	@echo "Full chain (assumes raw/interim inputs already on disk):"
	@echo "  make all / make pipeline  [data_process -> validate_merged -> features -> eda -> models]"

# ---------------------------------------------------------------------------
# Setup & quality gates
# ---------------------------------------------------------------------------
install:
	poetry install

lint:
	$(POETRY) ruff check .

lint-fix:
	$(POETRY) ruff check . --fix

format:
	$(POETRY) ruff format .

test:
	$(POETRY) pytest tests/ -v

test-cov:
	$(POETRY) pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

ci: lint test

# ---------------------------------------------------------------------------
# 1 — Acquire raw data
# ---------------------------------------------------------------------------
data_download:
	$(PYTHON) src/data/download_raw_data/acquire_airline_metadata.py
	$(PYTHON) src/data/download_raw_data/acquire_airport_metadata.py
	$(PYTHON) src/data/download_raw_data/download_flights_data.py
	$(PYTHON) src/data/download_raw_data/download_weather_iem_chunked.py

# ---------------------------------------------------------------------------
# 2 — Interim processing
# ---------------------------------------------------------------------------
data_combine_flights:
	$(PYTHON) src/data/interim_processing/combine_flights_ds.py

data_merge_weather:
	$(PYTHON) src/data/interim_processing/merge_weather.py

# ---------------------------------------------------------------------------
# 3 — Interim validation (exploratory / QA scripts)
# ---------------------------------------------------------------------------
validate_flights:
	$(PYTHON) src/data/validate_flights_ds.py

validate_weather:
	$(PYTHON) src/data/validate_weather_ds.py

# ---------------------------------------------------------------------------
# 4 — Final integration + cleaning + merged validation
# ---------------------------------------------------------------------------
data_merge_sources:
	$(PYTHON) src/data/final_processed/merge_data_sources.py

data_clean:
	$(PYTHON) src/data/clean_merged_dataset.py

validate_merged:
	$(PYTHON) src/data/validate_merged.py

validate_serve:
	$(POETRY) python -m http.server 8080

# Ordered: combine CSVs (optional if HF parquet used) -> weather -> merge sources -> clean
data_process: data_combine_flights data_merge_weather data_merge_sources data_clean

validate_all: validate_flights validate_weather validate_merged

# ---------------------------------------------------------------------------
# 5 — Features ^(train/valid/test artifacts under data/processed/^)
# ---------------------------------------------------------------------------
features:
	$(PYTHON) src/features/feature_pipeline.py

features_granular:
	$(PYTHON) src/features/preprocess.py
	$(PYTHON) src/features/feature_engineering.py
	$(PYTHON) src/features/feature_selection.py

# ---------------------------------------------------------------------------
# 6 — EDA ^(non-interactive; figures to reports/figures/^)
# ---------------------------------------------------------------------------
eda:
	$(PYTHON) src/notebooks/eda.py --data-dir data/processed --fig-dir reports/figures

# ---------------------------------------------------------------------------
# 7 — Models ^(MLflow logging inside model_training / model_comparison^)
# ---------------------------------------------------------------------------
models:
	$(PYTHON) src/models/model_pipeline.py

models_granular:
	$(PYTHON) src/models/class_weights.py
	$(PYTHON) src/models/model_training.py
	$(PYTHON) src/models/model_comparison.py
	$(PYTHON) src/models/hyperparameter_tuning.py

models_weights:
	$(PYTHON) src/models/model_weight_comparison.py

models_catboost:
	$(PYTHON) src/models/train_catboost_only.py

models_ensemble:
	$(PYTHON) src/models/stage3_ensemble.py

mlflow_ui:
	$(POETRY) mlflow ui --host 127.0.0.1 --port 5000

# ---------------------------------------------------------------------------
# Full pipeline (process -> validate merged -> features -> EDA -> train)
# ---------------------------------------------------------------------------
all: data_process validate_merged features eda models
	@echo Full pipeline finished.

pipeline: all
