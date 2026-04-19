"""
Module 5: Model Monitoring with Evidently 0.7.x

Uses evidently.legacy for the Report API.
Detects data drift between reference (Jan 2023) and
current (Feb 2023) production data.
"""

import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report
from sklearn.feature_extraction import DictVectorizer

# ── Config ────────────────────────────────────────────────────────────────────
TRACKING_URI = "sqlite:///02-experiment-tracking/mlflow.db"
REGISTRY_NAME = "nyc-taxi-duration-predictor"
STAGE = "Staging"
CATEGORICAL = ["PULocationID", "DOLocationID"]
NUMERICAL = ["trip_distance", "duration"]
FEATURES = CATEGORICAL + NUMERICAL
OUTPUT_DIR = "05-monitoring/dashboards"
SAMPLE_SIZE = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load & prepare data ───────────────────────────────────────────────────────
def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[CATEGORICAL] = df[CATEGORICAL].astype(str)
    return df[FEATURES]


print("Loading reference data (Jan 2023)...")
df_reference = load_and_prepare("data/raw/yellow_tripdata_2023-01.parquet")
df_reference = df_reference.sample(SAMPLE_SIZE, random_state=42)

print("Loading current data (Feb 2023)...")
df_current = load_and_prepare("data/raw/yellow_tripdata_2023-02.parquet")
df_current = df_current.sample(SAMPLE_SIZE, random_state=42)

print(f"Reference: {df_reference.shape}  |  Current: {df_current.shape}")


# ── Load model & add predictions ──────────────────────────────────────────────
print("\nLoading model from registry...")
mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}/{STAGE}")

# Fit vectorizer on FULL training data so all location IDs are seen
print("Fitting vectorizer on full training data...")
df_full = load_and_prepare("data/raw/yellow_tripdata_2023-01.parquet")
dv = DictVectorizer()
dv.fit(df_full[CATEGORICAL + ["trip_distance"]].to_dict(orient="records"))
print(f"Vectorizer fitted — {len(dv.feature_names_)} features ✅")

X_ref = dv.transform(
    df_reference[CATEGORICAL + ["trip_distance"]].to_dict(orient="records")
)
X_cur = dv.transform(
    df_current[CATEGORICAL + ["trip_distance"]].to_dict(orient="records")
)

df_reference = df_reference.copy()
df_current = df_current.copy()
df_reference["prediction"] = model.predict(X_ref)
df_current["prediction"] = model.predict(X_cur)

print(f"Reference predictions — mean: {df_reference['prediction'].mean():.2f} min")
print(f"Current predictions   — mean: {df_current['prediction'].mean():.2f} min")


# ── Evidently drift report ────────────────────────────────────────────────────
print("\nGenerating Evidently drift report...")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_reference, current_data=df_current)

html_path = os.path.join(OUTPUT_DIR, "drift_report.html")
report.save_html(html_path)
print(f"HTML report saved → {html_path} ✅")

json_path = os.path.join(OUTPUT_DIR, "drift_summary.json")
report_dict = report.as_dict()
with open(json_path, "w") as f:
    json.dump(report_dict, f, indent=2, default=str)
print(f"JSON summary saved → {json_path} ✅")


# ── Print drift summary ───────────────────────────────────────────────────────
print("\n── Drift Summary ──────────────────────────────────────────")
try:
    for metric in report_dict.get("metrics", []):
        result = metric.get("result", {})

        # Dataset-level summary
        if "dataset_drift" in result:
            drifted = result["dataset_drift"]
            n = result.get("number_of_drifted_columns", "?")
            total = result.get("number_of_columns", "?")
            share = result.get("share_of_drifted_columns", 0)
            print(f"Dataset drift detected : {drifted}")
            print(f"Drifted columns        : {n}/{total} ({share:.1%})\n")

        # Per-column breakdown
        if "drift_by_columns" in result:
            print(f"{'Column':<20}  {'Drift?':>8}  {'p-value':>10}  {'Score':>10}")
            print("-" * 55)
            for col, stats in result["drift_by_columns"].items():
                flag = "⚠️  YES" if stats.get("drift_detected") else "  no"
                p = stats.get("p_value", "-")
                s = stats.get("drift_score", "-")
                p_str = f"{p:.4f}" if isinstance(p, float) else str(p)
                s_str = f"{s:.4f}" if isinstance(s, float) else str(s)
                print(f"{col:<20}  {flag:>8}  {p_str:>10}  {s_str:>10}")
except Exception as e:
    print(f"Could not parse summary details: {e}")
    print("Full results available in drift_report.html")

print("\n✅ Module 5 monitoring complete!")
print(f"Open {html_path} in your browser to view the full interactive report.")
