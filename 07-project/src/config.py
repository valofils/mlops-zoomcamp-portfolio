"""Centralised config for the house price capstone project."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "AmesHousing.csv")
TRACKING_URI = (
    "sqlite:////workspaces/mlops-zoomcamp-portfolio/02-experiment-tracking/mlflow.db"
)
EXPERIMENT_NAME = "house-price-prediction"
REGISTRY_NAME = "house-price-predictor"

CATEGORICAL_FEATURES = [
    "MSZoning",
    "Street",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "Neighborhood",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "HeatingQC",
    "CentralAir",
    "KitchenQual",
    "GarageType",
    "SaleCondition",
]

NUMERICAL_FEATURES = [
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "GrLivArea",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "GarageCars",
    "GarageArea",
]

TARGET = "SalePrice"
TEST_SIZE = 0.2
RANDOM_STATE = 42
