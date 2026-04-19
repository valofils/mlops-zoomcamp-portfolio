# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt

# ── Data ──────────────────────────────────────────────────────────────────────
download-taxi:
	mkdir -p data/raw
	wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet -P data/raw/
	wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet -P data/raw/

download-ames:
	python - << 'PYEOF'
from sklearn.datasets import fetch_openml
import os
os.makedirs("07-project/data", exist_ok=True)
df = fetch_openml(name="house_prices", as_frame=True, parser="auto").frame
df.to_csv("07-project/data/AmesHousing.csv", index=False)
print(f"Downloaded: {df.shape[0]} rows x {df.shape[1]} columns")
PYEOF

download-all: download-taxi download-ames

# ── Training ──────────────────────────────────────────────────────────────────
train-taxi:
	python 02-experiment-tracking/scripts/train_with_mlflow.py

train-pipeline:
	python 03-orchestration/pipelines/training_pipeline.py

train-capstone:
	python 07-project/src/train.py

# ── Serving ───────────────────────────────────────────────────────────────────
serve-taxi:
	python 04-deployment/web-service/predict.py

serve-capstone:
	python 07-project/src/predict_api.py

# ── Monitoring ────────────────────────────────────────────────────────────────
monitor:
	python 05-monitoring/scripts/monitoring.py

# ── MLflow UI ─────────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///02-experiment-tracking/mlflow.db --host 0.0.0.0 --port 5000

# ── Quality & tests ───────────────────────────────────────────────────────────
quality:
	black .
	isort .

test:
	pytest 06-best-practices/tests/ 07-project/tests/ -v

test-all:
	pytest 06-best-practices/tests/ 07-project/tests/ -v --tb=short

# ── Pre-commit ────────────────────────────────────────────────────────────────
pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

# ── Full pipeline (capstone) ──────────────────────────────────────────────────
run-all: download-ames train-capstone monitor test
	@echo "✅ Full pipeline complete"

.PHONY: setup download-taxi download-ames download-all \
        train-taxi train-pipeline train-capstone \
        serve-taxi serve-capstone monitor mlflow-ui \
        quality test test-all pre-commit-install pre-commit-run run-all
