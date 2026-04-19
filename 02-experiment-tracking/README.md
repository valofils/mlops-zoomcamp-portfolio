# Module 2: Experiment Tracking & Model Management

## What This Module Covers

- Why experiment tracking matters
- MLflow: tracking runs, logging metrics/params/artifacts
- Model registry: staging, production, archiving
- Comparing experiments and selecting the best model

## Tools

- **MLflow** — experiment tracking and model registry

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_mlflow_intro.ipynb` | Getting started with MLflow tracking |
| `notebooks/02_model_registry.ipynb` | Registering and managing models |
| `notebooks/03_hyperparameter_tuning.ipynb` | Tracking hyperparameter search |

## Running MLflow UI

```bash
cd 02-experiment-tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

## Key Concepts

- **Run**: a single execution of your training script
- **Experiment**: a group of related runs
- **Artifact**: files saved with a run (models, plots, data)
- **Model Registry**: versioned store for production-ready models
- **Model Stage**: None → Staging → Production → Archived

## Notes & Learnings

_Add your notes here as you work through the module._
