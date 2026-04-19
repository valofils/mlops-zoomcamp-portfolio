"""
Capstone: House Price Prediction — Flask API
Loads the registered model from MLflow and serves predictions.
"""

import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.feature_extraction import DictVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    NUMERICAL_FEATURES,
    REGISTRY_NAME,
    TRACKING_URI,
)

# ── Load model & vectorizer on startup ────────────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)
print(f"Loading model: {REGISTRY_NAME}/Staging...")
model = mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}/Staging")
print(f"Model type: {type(model).__name__} ✅")

# Rebuild vectorizer from training data
print("Fitting vectorizer...")
df_ref = pd.read_csv(DATA_PATH)
df_ref = df_ref.dropna(subset=["SalePrice"])
for col in CATEGORICAL_FEATURES:
    if col in df_ref.columns:
        df_ref[col] = df_ref[col].fillna("missing").astype(str)
for col in NUMERICAL_FEATURES:
    if col in df_ref.columns:
        df_ref[col] = df_ref[col].fillna(df_ref[col].median())

available = [
    f for f in CATEGORICAL_FEATURES + NUMERICAL_FEATURES if f in df_ref.columns
]
dv = DictVectorizer()
dv.fit(df_ref[available].to_dict(orient="records"))
print(f"Vectorizer ready — {len(dv.feature_names_)} features ✅")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

REQUIRED_FIELDS = [
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "Neighborhood",
    "GarageCars",
]


def prepare_features(house: dict) -> dict:
    features = {}
    for col in CATEGORICAL_FEATURES:
        features[col] = str(house.get(col, "missing"))
    for col in NUMERICAL_FEATURES:
        features[col] = float(house.get(col, 0))
    return features


def predict_price(features: dict) -> float:
    X = dv.transform([features])
    log_pred = model.predict(X)[0]
    return float(round(np.expm1(log_pred), 2))


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": REGISTRY_NAME,
            "model_type": type(model).__name__,
            "features": len(dv.feature_names_),
        }
    )


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    house = request.get_json()

    if not house:
        return jsonify({"error": "No JSON body provided"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in house]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    features = prepare_features(house)
    price = predict_price(features)

    return jsonify(
        {
            "predicted_sale_price_usd": price,
            "model": REGISTRY_NAME,
            "model_type": type(model).__name__,
        }
    )


@app.route("/features", methods=["GET"])
def list_features():
    return jsonify(
        {
            "categorical": CATEGORICAL_FEATURES,
            "numerical": NUMERICAL_FEATURES,
            "required": REQUIRED_FIELDS,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9697, debug=False)
