validate:
	poetry run python validate_merged.py
	poetry run python -m http.server 8080
