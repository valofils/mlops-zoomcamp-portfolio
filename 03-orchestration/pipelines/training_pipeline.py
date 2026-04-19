import os
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# ── Config ────────────────────────────────────────────────────────────────────
TRACKING_URI = "sqlite:///02-experiment-tracking/mlflow.db"
EXPERIMENT_NAME = "nyc-taxi-duration"
REGISTRY_NAME = "nyc-taxi-duration-predictor"
CATEGORICAL = ["PULocationID", "DOLocationID"]
NUMERICAL = ["trip_distance"]
DATA_PATH = "data/raw"


# ── Tasks ─────────────────────────────────────────────────────────────────────
@task(name="load-data", retries=2, retry_delay_seconds=5)
def load_data(month: str) -> pd.DataFrame:
    logger = get_run_logger()
    path = f"{DATA_PATH}/yellow_tripdata_{month}.parquet"
    logger.info(f"Loading data from {path}")

    df = pd.read_parquet(path)
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[CATEGORICAL] = df[CATEGORICAL].astype(str)

    logger.info(f"Loaded {len(df):,} rows from {month}")
    return df


@task(name="build-features")
def build_features(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple:
    logger = get_run_logger()

    dv = DictVectorizer()
    X_train = dv.fit_transform(
        df_train[CATEGORICAL + NUMERICAL].to_dict(orient="records")
    )
    X_val = dv.transform(df_val[CATEGORICAL + NUMERICAL].to_dict(orient="records"))
    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    logger.info(f"Features built — train: {X_train.shape}, val: {X_val.shape}")
    return dv, X_train, X_val, y_train, y_val


@task(name="train-model")
def train_model(
    model_name: str,
    model,
    params: dict,
    X_train,
    y_train,
    X_val,
    y_val,
) -> dict:
    logger = get_run_logger()

    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_params({**params, "model_type": model_name})
        model.fit(X_train, y_train)

        rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        rmse_val = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

        mlflow.log_metric("rmse_train", round(rmse_train, 4))
        mlflow.log_metric("rmse_val", round(rmse_val, 4))
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"{model_name}: train={rmse_train:.3f} val={rmse_val:.3f}")

    return {"name": model_name, "run_id": run_id, "rmse_val": rmse_val, "model": model}


@task(name="select-best-model")
def select_best_model(results: list) -> dict:
    logger = get_run_logger()
    best = min(results, key=lambda x: x["rmse_val"])
    logger.info(f"Best model: {best['name']} (val RMSE={best['rmse_val']:.4f})")
    return best


@task(name="register-model")
def register_model(best: dict) -> str:
    logger = get_run_logger()
    client = MlflowClient()

    model_uri = f"runs:/{best['run_id']}/model"
    result = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)

    client.transition_model_version_stage(
        name=REGISTRY_NAME,
        version=result.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    logger.info(f"Registered {REGISTRY_NAME} v{result.version} → Staging")
    return result.version


@task(name="save-results-plot")
def save_results_plot(results: list) -> None:
    logger = get_run_logger()
    df = pd.DataFrame(
        [{"model": r["name"], "rmse_val": r["rmse_val"]} for r in results]
    ).sort_values("rmse_val")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(df["model"], df["rmse_val"], color="steelblue")
    ax.set_xlabel("Val RMSE (minutes)")
    ax.set_title("Prefect Pipeline — Model Comparison")
    ax.axvline(df["rmse_val"].min(), color="coral", linestyle="--", label="Best")
    ax.legend()
    plt.tight_layout()

    os.makedirs("data/processed", exist_ok=True)
    plt.savefig("data/processed/prefect_model_comparison.png", dpi=100)
    logger.info("Plot saved to data/processed/prefect_model_comparison.png")


# ── Flow ──────────────────────────────────────────────────────────────────────
@flow(name="nyc-taxi-training-pipeline", log_prints=True)
def training_pipeline(
    train_month: str = "2023-01",
    val_month: str = "2023-02",
):
    logger = get_run_logger()
    logger.info(f"Starting pipeline: train={train_month}, val={val_month}")

    # Set up MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    df_train = load_data(train_month)
    df_val = load_data(val_month)

    # Build features
    dv, X_train, X_val, y_train, y_val = build_features(df_train, df_val)

    # Train all models
    experiments = [
        ("LinearRegression", LinearRegression(), {}),
        ("Lasso_alpha_0.01", Lasso(alpha=0.01), {"alpha": 0.01}),
        ("Lasso_alpha_0.1", Lasso(alpha=0.1), {"alpha": 0.1}),
        ("Ridge_alpha_0.1", Ridge(alpha=0.1), {"alpha": 0.1}),
        ("Ridge_alpha_1.0", Ridge(alpha=1.0), {"alpha": 1.0}),
    ]

    results = []
    with mlflow.start_run(run_name="prefect-pipeline-run"):
        for name, model, params in experiments:
            result = train_model(name, model, params, X_train, y_train, X_val, y_val)
            results.append(result)

    # Select and register best
    best = select_best_model(results)
    version = register_model(best)
    save_results_plot(results)

    logger.info(f"Pipeline complete — best: {best['name']} v{version}")
    return best


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    training_pipeline()
