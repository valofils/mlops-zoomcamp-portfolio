import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "nyc-taxi-duration"
REGISTRY_NAME = "nyc-taxi-duration-predictor"
CATEGORICAL = ["PULocationID", "DOLocationID"]
NUMERICAL = ["trip_distance"]

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"Tracking URI : {mlflow.get_tracking_uri()}")
print(f"Experiment   : {EXPERIMENT_NAME}\n")


# ── Data preparation ──────────────────────────────────────────────────────────
def load_and_prepare(path):
    df = pd.read_parquet(path)
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[CATEGORICAL] = df[CATEGORICAL].astype(str)
    return df


print("Loading data...")
df_train = load_and_prepare("../data/raw/yellow_tripdata_2023-01.parquet")
df_val = load_and_prepare("../data/raw/yellow_tripdata_2023-02.parquet")

dv = DictVectorizer()
X_train = dv.fit_transform(df_train[CATEGORICAL + NUMERICAL].to_dict(orient="records"))
X_val = dv.transform(df_val[CATEGORICAL + NUMERICAL].to_dict(orient="records"))
y_train = df_train["duration"].values
y_val = df_val["duration"].values
print(f"Train: {X_train.shape}  |  Val: {X_val.shape}\n")


# ── Training helper ───────────────────────────────────────────────────────────
def log_run(model_name, model, params):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({**params, "model_type": model_name})
        model.fit(X_train, y_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        rmse_val = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        mlflow.log_metric("rmse_train", round(rmse_train, 4))
        mlflow.log_metric("rmse_val", round(rmse_val, 4))
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = mlflow.active_run().info.run_id
        print(
            f"{model_name:<25}  train={rmse_train:.3f}  val={rmse_val:.3f}  id={run_id[:8]}"
        )
        return run_id, rmse_val


# ── Experiments ───────────────────────────────────────────────────────────────
experiments = [
    ("LinearRegression", LinearRegression(), {}),
    ("Lasso_alpha_0.01", Lasso(alpha=0.01), {"alpha": 0.01}),
    ("Lasso_alpha_0.1", Lasso(alpha=0.1), {"alpha": 0.1}),
    ("Lasso_alpha_1.0", Lasso(alpha=1.0), {"alpha": 1.0}),
    ("Ridge_alpha_0.1", Ridge(alpha=0.1), {"alpha": 0.1}),
    ("Ridge_alpha_1.0", Ridge(alpha=1.0), {"alpha": 1.0}),
    ("Ridge_alpha_10.0", Ridge(alpha=10.0), {"alpha": 10.0}),
]

print(f"Running {len(experiments)} experiments...\n")
run_results = []
for name, model, params in experiments:
    run_id, rmse_v = log_run(name, model, params)
    run_results.append({"name": name, "run_id": run_id, "rmse_val": rmse_v})

# ── Compare runs ──────────────────────────────────────────────────────────────
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id], order_by=["metrics.rmse_val ASC"]
)

print(f"\nTotal runs logged: {len(runs)}\n")
print(f"{'Model':<25}  {'Val RMSE':>10}  {'Train RMSE':>12}")
print("-" * 55)
for r in runs:
    name = r.data.params.get("model_type", "unknown")
    rmse_v = r.data.metrics.get("rmse_val", 0)
    rmse_tr = r.data.metrics.get("rmse_train", 0)
    print(f"{name:<25}  {rmse_v:>10.4f}  {rmse_tr:>12.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
records = [
    {
        "model": r.data.params.get("model_type", "?"),
        "rmse_val": r.data.metrics.get("rmse_val", 0),
        "rmse_train": r.data.metrics.get("rmse_train", 0),
    }
    for r in runs
]
results_df = pd.DataFrame(records).sort_values("rmse_val")

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(results_df))
w = 0.35
ax.bar(x - w / 2, results_df["rmse_train"], w, label="Train RMSE", color="steelblue")
ax.bar(x + w / 2, results_df["rmse_val"], w, label="Val RMSE", color="coral")
ax.set_xticks(x)
ax.set_xticklabels(results_df["model"], rotation=30, ha="right")
ax.set_ylabel("RMSE (minutes)")
ax.set_title("MLflow Experiment Comparison — NYC Taxi Duration")
ax.legend()
ax.set_ylim(0, 15)
plt.tight_layout()
plt.savefig("../data/processed/mlflow_experiment_comparison.png", dpi=100)
print("\nPlot saved ✅")

# ── Register best model ───────────────────────────────────────────────────────
best_run = runs[0]
best_run_id = best_run.info.run_id
best_name = best_run.data.params.get("model_type", "unknown")
best_rmse = best_run.data.metrics["rmse_val"]

print(f"\nBest model   : {best_name}")
print(f"Best val RMSE: {best_rmse:.4f}")

model_uri = f"runs:/{best_run_id}/model"
result = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)
print(f"Registered   : {result.name} v{result.version} ✅")

client.transition_model_version_stage(
    name=REGISTRY_NAME,
    version=result.version,
    stage="Staging",
    archive_existing_versions=True,
)
print(f"Staged       : v{result.version} → Staging ✅")

# ── Verify staged model ───────────────────────────────────────────────────────
staged = mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}/Staging")
preds = staged.predict(X_val[:5])
print("\nSample predictions vs actuals (minutes):")
for p, a in zip(preds, y_val[:5]):
    print(f"  predicted={p:.1f}  actual={a:.1f}  diff={abs(p-a):.1f}")

print("\n✅ Module 2 complete!")
