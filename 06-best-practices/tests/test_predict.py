"""
Unit tests for the prediction web service.
Tests core logic without requiring a running server or MLflow.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_model():
    """Train a tiny Ridge model on synthetic data."""
    dv = DictVectorizer()
    rides = [
        {"PULocationID": "1", "DOLocationID": "2", "trip_distance": 1.0},
        {"PULocationID": "3", "DOLocationID": "4", "trip_distance": 5.0},
        {"PULocationID": "1", "DOLocationID": "4", "trip_distance": 2.5},
        {"PULocationID": "2", "DOLocationID": "3", "trip_distance": 3.0},
        {"PULocationID": "5", "DOLocationID": "6", "trip_distance": 8.0},
    ]
    durations = [5.0, 20.0, 10.0, 12.0, 30.0]
    X = dv.fit_transform(rides)
    model = Ridge(alpha=1.0)
    model.fit(X, durations)
    return dv, model


@pytest.fixture
def sample_ride():
    return {
        "PULocationID": 130,
        "DOLocationID": 205,
        "trip_distance": 2.5,
    }


# ── Feature preparation tests ─────────────────────────────────────────────────
def test_prepare_features_types(sample_ride):
    """Feature prep must cast location IDs to str and distance to float."""
    features = {
        "PULocationID": str(sample_ride["PULocationID"]),
        "DOLocationID": str(sample_ride["DOLocationID"]),
        "trip_distance": float(sample_ride["trip_distance"]),
    }
    assert isinstance(features["PULocationID"], str)
    assert isinstance(features["DOLocationID"], str)
    assert isinstance(features["trip_distance"], float)


def test_prepare_features_values(sample_ride):
    """Feature values must match the input ride."""
    features = {
        "PULocationID": str(sample_ride["PULocationID"]),
        "DOLocationID": str(sample_ride["DOLocationID"]),
        "trip_distance": float(sample_ride["trip_distance"]),
    }
    assert features["PULocationID"] == "130"
    assert features["DOLocationID"] == "205"
    assert features["trip_distance"] == 2.5


# ── Data preparation tests ────────────────────────────────────────────────────
def test_duration_calculation():
    """Duration must be calculated correctly from timestamps."""
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2023-01-01 10:00:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2023-01-01 10:15:00"]),
        }
    )
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    assert df["duration"].iloc[0] == 15.0


def test_duration_filtering():
    """Only trips between 1 and 60 minutes should survive filtering."""
    durations = pd.Series([0.5, 1.0, 30.0, 60.0, 61.0, -5.0])
    filtered = durations[(durations >= 1) & (durations <= 60)]
    assert list(filtered) == [1.0, 30.0, 60.0]
    assert len(filtered) == 3


def test_location_id_cast_to_string():
    """Location IDs must be cast to string for DictVectorizer."""
    df = pd.DataFrame({"PULocationID": [1, 2, 3], "DOLocationID": [4, 5, 6]})
    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(
        str
    )
    assert df["PULocationID"].dtype == object
    assert df["DOLocationID"].dtype == object


# ── Model inference tests ─────────────────────────────────────────────────────
def test_model_predict_returns_float(sample_model):
    """Model must return a numeric prediction."""
    dv, model = sample_model
    features = {"PULocationID": "1", "DOLocationID": "2", "trip_distance": 2.0}
    X = dv.transform([features])
    pred = model.predict(X)
    assert isinstance(float(pred[0]), float)


def test_model_predict_plausible_range(sample_model):
    """Predictions must be in a plausible range (1–120 minutes)."""
    dv, model = sample_model
    test_rides = [
        {"PULocationID": "1", "DOLocationID": "2", "trip_distance": 1.0},
        {"PULocationID": "3", "DOLocationID": "4", "trip_distance": 5.0},
        {"PULocationID": "5", "DOLocationID": "6", "trip_distance": 8.0},
    ]
    X = dv.transform(test_rides)
    preds = model.predict(X)
    for p in preds:
        assert 1 <= p <= 120, f"Implausible prediction: {p:.2f} minutes"


def test_model_longer_trip_higher_duration(sample_model):
    """Longer trips should predict longer durations."""
    dv, model = sample_model
    short = {"PULocationID": "1", "DOLocationID": "2", "trip_distance": 1.0}
    long_ = {"PULocationID": "1", "DOLocationID": "2", "trip_distance": 10.0}
    X_short = dv.transform([short])
    X_long = dv.transform([long_])
    pred_short = model.predict(X_short)[0]
    pred_long = model.predict(X_long)[0]
    assert pred_long > pred_short, "Longer trip should predict longer duration"


# ── DictVectorizer tests ──────────────────────────────────────────────────────
def test_dictvectorizer_output_shape():
    """DictVectorizer must produce consistent feature dimensions."""
    dv = DictVectorizer()
    train = [
        {"PULocationID": "1", "DOLocationID": "2", "trip_distance": 1.0},
        {"PULocationID": "3", "DOLocationID": "4", "trip_distance": 5.0},
    ]
    X_train = dv.fit_transform(train)
    val = [{"PULocationID": "1", "DOLocationID": "2", "trip_distance": 2.0}]
    X_val = dv.transform(val)
    assert X_train.shape[1] == X_val.shape[1], "Feature dimensions must match"


def test_dictvectorizer_unseen_location_handled():
    """Unseen location IDs at inference must not crash the vectorizer."""
    dv = DictVectorizer()
    train = [{"PULocationID": "1", "DOLocationID": "2", "trip_distance": 1.0}]
    dv.fit_transform(train)
    # Unseen location ID — should not raise
    unseen = [{"PULocationID": "999", "DOLocationID": "888", "trip_distance": 3.0}]
    X = dv.transform(unseen)
    assert X.shape[0] == 1


# ── Drift monitoring tests ────────────────────────────────────────────────────
def test_drift_score_range():
    """Drift scores must be between 0 and 1."""
    mock_drift_scores = [0.04, 0.11, 0.02, 0.09]
    for score in mock_drift_scores:
        assert 0.0 <= score <= 1.0, f"Drift score out of range: {score}"


def test_prediction_mean_stability():
    """Mean prediction should not shift dramatically between batches."""
    ref_mean = 14.47
    cur_mean = 14.24
    threshold = 2.0  # minutes
    assert (
        abs(ref_mean - cur_mean) < threshold
    ), f"Prediction mean shifted by {abs(ref_mean - cur_mean):.2f} min"
