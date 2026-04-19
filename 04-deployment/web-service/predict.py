import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction import DictVectorizer

# ── Config ────────────────────────────────────────────────────────────────────
TRACKING_URI = "sqlite:///../../02-experiment-tracking/mlflow.db"
REGISTRY_NAME = "nyc-taxi-duration-predictor"
STAGE = "Staging"
CATEGORICAL = ["PULocationID", "DOLocationID"]
NUMERICAL = ["trip_distance"]

# ── Load model & rebuild vectorizer on startup ────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)

print(f"Loading model: {REGISTRY_NAME}/{STAGE}...")
model = mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}/{STAGE}")
print(f"Model type: {type(model).__name__} ✅")

# Rebuild the DictVectorizer by fitting on a small reference dataset
# In production this would be saved as an artifact — Module 5 fixes this
print("Fitting reference vectorizer...")
df_ref = pd.read_parquet("../../data/raw/yellow_tripdata_2023-01.parquet")
df_ref["duration"] = (
    df_ref.tpep_dropoff_datetime - df_ref.tpep_pickup_datetime
).dt.total_seconds() / 60
df_ref = df_ref[(df_ref.duration >= 1) & (df_ref.duration <= 60)].copy()
df_ref[CATEGORICAL] = df_ref[CATEGORICAL].astype(str)

dv = DictVectorizer()
dv.fit(df_ref[CATEGORICAL + NUMERICAL].to_dict(orient="records"))
print(f"Vectorizer ready — {len(dv.feature_names_)} features ✅")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


def prepare_features(ride: dict) -> dict:
    return {
        "PULocationID": str(ride["PULocationID"]),
        "DOLocationID": str(ride["DOLocationID"]),
        "trip_distance": float(ride["trip_distance"]),
    }


def predict(features: dict) -> float:
    X = dv.transform([features])
    pred = model.predict(X)
    return float(round(pred[0], 2))


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": REGISTRY_NAME,
            "stage": STAGE,
            "model_type": type(model).__name__,
        }
    )


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()

    if not ride:
        return jsonify({"error": "No JSON body provided"}), 400

    required = ["PULocationID", "DOLocationID", "trip_distance"]
    missing = [f for f in required if f not in ride]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    features = prepare_features(ride)
    duration = predict(features)

    return jsonify(
        {
            "predicted_duration_minutes": duration,
            "model": REGISTRY_NAME,
            "stage": STAGE,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=False)
