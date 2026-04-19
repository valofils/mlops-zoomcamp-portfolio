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

## Reproducing This Project

### 1. Download the dataset

The Ames Housing dataset is fetched from OpenML automatically:

```python
from sklearn.datasets import fetch_openml
import os

os.makedirs("07-project/data", exist_ok=True)
ames = fetch_openml(name="house_prices", as_frame=True, parser="auto")
ames.frame.to_csv("07-project/data/AmesHousing.csv", index=False)
```

Or run the convenience script:

```bash
python - << 'EOF'
from sklearn.datasets import fetch_openml
import os
os.makedirs("07-project/data", exist_ok=True)
df = fetch_openml(name="house_prices", as_frame=True, parser="auto").frame
df.to_csv("07-project/data/AmesHousing.csv", index=False)
print(f"Downloaded: {df.shape[0]} rows x {df.shape[1]} columns")
EOF
```

### 2. Train the model

```bash
python 07-project/src/train.py
```

### 3. Start the prediction API

```bash
python 07-project/src/predict_api.py
```

### 4. Run tests

```bash
pytest 07-project/tests/ -v
```

### 5. Example prediction request

```bash
curl -X POST http://localhost:9697/predict \
  -H "Content-Type: application/json" \
  -d '{
    "GrLivArea": 1500,
    "OverallQual": 7,
    "YearBuilt": 2000,
    "Neighborhood": "CollgCr",
    "GarageCars": 2,
    "LotArea": 9500
  }'
```

Expected response:
```json
{
  "model": "house-price-predictor",
  "model_type": "GradientBoostingRegressor",
  "predicted_sale_price_usd": 198432.50
}
```
