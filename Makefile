.PHONY: data_download data_process features models all validate

# 1. Download all raw data
data_download:
	poetry run python src/data/download_raw_data/acquire_airline_metadata.py
	poetry run python src/data/download_raw_data/acquire_airport_metadata.py
	poetry run python src/data/download_raw_data/download_flights_data.py
	poetry run python src/data/download_raw_data/download_weather_iem_chunked.py

# 2. Process and merge the data
data_process:
	poetry run python src/data/interim_processing/combine_flights_ds.py
	poetry run python src/data/interim_processing/merge_weather.py
	poetry run python src/data/final_processed/merge_data_sources.py
	poetry run python src/data/clean_merged_dataset.py

# 3. Validation (Split into run and serve)
validate_run:
	poetry run python src/data/validate_merged.py

validate_serve:
	poetry run python -m http.server 8080

# 4. Preprocessing, Engineering, and Selection
features:
	poetry run python src/features/preprocess.py
	poetry run python src/features/feature_engineering.py
	poetry run python src/features/feature_selection.py

# 5. Train, Compare, and Tune Models
models:
	poetry run python src/models/class_weights.py
	poetry run python src/models/model_training.py
	poetry run python src/models/model_comparison.py
	poetry run python src/models/hyperparameter_tuning.py

# The master pipeline command 
all: data_download data_process validate_run features models
	@echo "Full pipeline executed successfully!"