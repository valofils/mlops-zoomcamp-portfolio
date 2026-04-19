"""
Unit tests for the house price prediction capstone.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import r2_score


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_houses():
    return [
        {
            "GrLivArea": 1000,
            "OverallQual": 5,
            "YearBuilt": 1980,
            "Neighborhood": "NAmes",
            "GarageCars": 1,
            "LotArea": 8000,
        },
        {
            "GrLivArea": 2000,
            "OverallQual": 8,
            "YearBuilt": 2005,
            "Neighborhood": "NridgHt",
            "GarageCars": 2,
            "LotArea": 12000,
        },
        {
            "GrLivArea": 1500,
            "OverallQual": 6,
            "YearBuilt": 1995,
            "Neighborhood": "CollgCr",
            "GarageCars": 2,
            "LotArea": 9500,
        },
    ]


@pytest.fixture
def trained_model(sample_houses):
    dv = DictVectorizer()
    X = dv.fit_transform(sample_houses)
    y = np.log1p([120000, 280000, 180000])
    model = GradientBoostingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return dv, model


# ── Feature preparation tests ─────────────────────────────────────────────────
def test_categorical_features_cast_to_string():
    house = {"Neighborhood": 123, "MSZoning": None}
    result = {
        "Neighborhood": str(house.get("Neighborhood", "missing")),
        "MSZoning": str(house.get("MSZoning", "missing")),
    }
    assert result["Neighborhood"] == "123"
    assert result["MSZoning"] == "None"


def test_numerical_features_cast_to_float():
    house = {"GrLivArea": "1500", "OverallQual": 7}
    result = {
        "GrLivArea": float(house.get("GrLivArea", 0)),
        "OverallQual": float(house.get("OverallQual", 0)),
    }
    assert result["GrLivArea"] == 1500.0
    assert isinstance(result["OverallQual"], float)


def test_missing_fields_default_to_zero():
    house = {"GrLivArea": 1200}
    numerical = ["GrLivArea", "OverallQual", "YearBuilt"]
    result = {col: float(house.get(col, 0)) for col in numerical}
    assert result["GrLivArea"] == 1200.0
    assert result["OverallQual"] == 0.0
    assert result["YearBuilt"] == 0.0


# ── Data preparation tests ────────────────────────────────────────────────────
def test_log_transform_target():
    prices = [100000, 200000, 300000]
    log_prices = np.log1p(prices)
    recovered = np.expm1(log_prices)
    np.testing.assert_allclose(recovered, prices, rtol=1e-5)


def test_missing_numerical_filled_with_median():
    df = pd.DataFrame({"GrLivArea": [1000, 1500, np.nan, 2000]})
    df["GrLivArea"] = df["GrLivArea"].fillna(df["GrLivArea"].median())
    assert df["GrLivArea"].isna().sum() == 0
    assert df["GrLivArea"].iloc[2] == 1500.0


def test_missing_categorical_filled_with_missing():
    df = pd.DataFrame({"Neighborhood": ["NAmes", None, "CollgCr"]})
    df["Neighborhood"] = df["Neighborhood"].fillna("missing").astype(str)
    assert df["Neighborhood"].iloc[1] == "missing"
    assert df["Neighborhood"].isna().sum() == 0


# ── Model inference tests ─────────────────────────────────────────────────────
def test_model_predicts_float(trained_model):
    dv, model = trained_model
    house = {
        "GrLivArea": 1200,
        "OverallQual": 6,
        "YearBuilt": 1990,
        "Neighborhood": "NAmes",
        "GarageCars": 1,
        "LotArea": 8500,
    }
    X = dv.transform([house])
    pred = model.predict(X)
    assert isinstance(float(pred[0]), float)


def test_model_predicts_plausible_price(trained_model):
    dv, model = trained_model
    house = {
        "GrLivArea": 1500,
        "OverallQual": 6,
        "YearBuilt": 1995,
        "Neighborhood": "CollgCr",
        "GarageCars": 2,
        "LotArea": 9500,
    }
    X = dv.transform([house])
    log_pred = model.predict(X)[0]
    price = np.expm1(log_pred)
    assert 50000 <= price <= 1000000, f"Implausible price: ${price:,.0f}"


def test_larger_house_higher_price(trained_model):
    dv, model = trained_model
    small = {
        "GrLivArea": 800,
        "OverallQual": 5,
        "YearBuilt": 1980,
        "Neighborhood": "NAmes",
        "GarageCars": 1,
        "LotArea": 7000,
    }
    large = {
        "GrLivArea": 2500,
        "OverallQual": 8,
        "YearBuilt": 2005,
        "Neighborhood": "NridgHt",
        "GarageCars": 3,
        "LotArea": 15000,
    }
    X_small = dv.transform([small])
    X_large = dv.transform([large])
    price_small = np.expm1(model.predict(X_small)[0])
    price_large = np.expm1(model.predict(X_large)[0])
    assert price_large > price_small


def test_dictvectorizer_consistent_dimensions(sample_houses):
    dv = DictVectorizer()
    X_train = dv.fit_transform(sample_houses[:2])
    X_val = dv.transform(sample_houses[2:])
    assert X_train.shape[1] == X_val.shape[1]


# ── Business logic tests ──────────────────────────────────────────────────────
def test_r2_above_threshold(trained_model):
    dv, model = trained_model
    houses = [
        {
            "GrLivArea": 1000,
            "OverallQual": 5,
            "YearBuilt": 1980,
            "Neighborhood": "NAmes",
            "GarageCars": 1,
            "LotArea": 8000,
        },
        {
            "GrLivArea": 2000,
            "OverallQual": 8,
            "YearBuilt": 2005,
            "Neighborhood": "NridgHt",
            "GarageCars": 2,
            "LotArea": 12000,
        },
        {
            "GrLivArea": 1500,
            "OverallQual": 6,
            "YearBuilt": 1995,
            "Neighborhood": "CollgCr",
            "GarageCars": 2,
            "LotArea": 9500,
        },
    ]
    y_true = np.log1p([120000, 280000, 180000])
    X = dv.transform(houses)
    y_pred = model.predict(X)
    r2 = r2_score(y_true, y_pred)
    assert r2 > 0.5, f"R2 too low: {r2:.3f}"


def test_prediction_within_reasonable_error(trained_model):
    dv, model = trained_model
    house = {
        "GrLivArea": 1000,
        "OverallQual": 5,
        "YearBuilt": 1980,
        "Neighborhood": "NAmes",
        "GarageCars": 1,
        "LotArea": 8000,
    }
    X = dv.transform([house])
    pred_price = np.expm1(model.predict(X)[0])
    actual_price = 120000
    error_pct = abs(pred_price - actual_price) / actual_price
    assert error_pct < 0.50, f"Error too large: {error_pct:.1%}"
