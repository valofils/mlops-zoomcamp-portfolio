# Capstone: House Price Prediction

End-to-end MLOps project predicting residential property sale prices
using the Ames Housing dataset (1,460 properties, 80 features).

## Problem Statement

Predict the sale price of a residential property given structural,
locational, and quality features. Accurate price estimates support
real estate valuation, mortgage underwriting, and investment decisions.

## Dataset

- Source: Ames Housing (OpenML)
- Size: 1,460 rows x 81 columns
- Target: SalePrice (continuous, USD)
- Features: 79 predictors covering lot size, neighborhood,
  building type, quality ratings, year built, and more

## MLOps Stack

| Layer | Tool |
|-------|------|
| Experiment tracking | MLflow |
| Orchestration | Prefect |
| Deployment | Flask REST API |
| Monitoring | Evidently |
| Testing | pytest |
| CI/CD | GitHub Actions |

## Quick Start

    pip install -r requirements.txt
    python 07-project/src/train.py
    python 07-project/src/predict_api.py
    pytest 07-project/tests/ -v
