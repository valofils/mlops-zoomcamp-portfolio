setup:
	pip install -r requirements.txt

mlflow-ui:
	cd 02-experiment-tracking && mlflow ui --backend-store-uri sqlite:///mlflow.db

quality:
	black .
	isort .
	pylint src/ --fail-under=7

test:
	pytest tests/ -v

pre-commit-install:
	pre-commit install

.PHONY: setup mlflow-ui quality test pre-commit-install
