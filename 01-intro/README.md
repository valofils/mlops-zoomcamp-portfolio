# Module 1: Introduction to MLOps

## What This Module Covers

- What MLOps is and why it matters
- The MLOps maturity model (levels 0-4)
- Setting up the development environment
- Running a baseline ML model (NY Taxi duration prediction)
- Understanding the gap between a notebook model and a production service

## Dataset

NY Taxi Trip Duration. Download Jan & Feb 2023 data:

```bash
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet -P ../data/raw/
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet -P ../data/raw/
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_baseline.ipynb` | Data prep, linear regression, evaluation |

## MLOps Maturity Model

| Level | Description |
|-------|-------------|
| 0 | No MLOps — notebooks, fully manual |
| 1 | DevOps but no MLOps — releases automated, no experiment tracking |
| 2 | Automated training — training pipeline, experiment tracking |
| 3 | Automated deployment — easy model deployment |
| 4 | Full MLOps — automated retraining and monitoring |

## Notes & Learnings

_Add your notes here as you work through the module._
