"""
Capstone: House Price Prediction — Training Script
Trains multiple models with MLflow tracking and registers the best one.
"""

import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    EXPERIMENT_NAME,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    REGISTRY_NAME,
    TARGET,
    TEST_SIZE,
    TRACKING_URI,
)

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Tracking URI: {TRACKING_URI}\n")


# ── Data loading & preparation ────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])

    # Fill missing categoricals with "missing"
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)

    # Fill missing numericals with median
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"After cleaning: {df.shape[0]} rows")
    return df


def build_features(df: pd.DataFrame, dv: DictVectorizer = None, fit: bool = False):
    features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    available = [f for f in features if f in df.columns]
    records = df[available].to_dict(orient="records")

    if fit:
        dv = DictVectorizer()
        X = dv.fit_transform(records)
    else:
        X = dv.transform(records)

    y = np.log1p(df[TARGET].values)  # log-transform target
    return dv, X, y


# ── Training helper ───────────────────────────────────────────────────────────
def log_run(name, model, params, X_tr, y_tr, X_v, y_v):
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params({**params, "model_type": name})
        mlflow.log_param("features", len(CATEGORICAL_FEATURES + NUMERICAL_FEATURES))

        model.fit(X_tr, y_tr)

        # Evaluate on log scale
        rmse_train = np.sqrt(mean_squared_error(y_tr, model.predict(X_tr)))
        rmse_val = np.sqrt(mean_squared_error(y_v, model.predict(X_v)))
        r2_val = r2_score(y_v, model.predict(X_v))

        # Evaluate on original scale (expm1 reverses log1p)
        pred_val_orig = np.expm1(model.predict(X_v))
        y_v_orig = np.expm1(y_v)
        rmse_val_orig = np.sqrt(mean_squared_error(y_v_orig, pred_val_orig))

        mlflow.log_metric("rmse_train_log", round(rmse_train, 4))
        mlflow.log_metric("rmse_val_log", round(rmse_val, 4))
        mlflow.log_metric("rmse_val_usd", round(rmse_val_orig, 2))
        mlflow.log_metric("r2_val", round(r2_val, 4))
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(
            f"{name:<35} rmse_log={rmse_val:.4f}  "
            f"rmse_usd=${rmse_val_orig:,.0f}  r2={r2_val:.4f}"
        )
        return run_id, rmse_val, r2_val


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    df_train, df_val = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(df_train)}  |  Val: {len(df_val)}\n")

    dv, X_train, y_train = build_features(df_train, fit=True)
    _, X_val, y_val = build_features(df_val, dv=dv, fit=False)
    print(f"Feature matrix: {X_train.shape}\n")

    experiments = [
        ("Ridge_alpha_1", Ridge(alpha=1.0), {"alpha": 1.0}),
        ("Ridge_alpha_10", Ridge(alpha=10.0), {"alpha": 10.0}),
        ("Lasso_alpha_0.001", Lasso(alpha=0.001), {"alpha": 0.001}),
        (
            "RandomForest_n50",
            RandomForestRegressor(
                n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
            ),
            {"n_estimators": 50, "max_depth": 10},
        ),
        (
            "GradientBoosting_n100",
            GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ),
            {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
        ),
    ]

    results = []
    print("Running experiments...\n")
    with mlflow.start_run(run_name="house-price-training"):
        for name, model, params in experiments:
            run_id, rmse_val, r2_val = log_run(
                name, model, params, X_train, y_train, X_val, y_val
            )
            results.append(
                {"name": name, "run_id": run_id, "rmse_val": rmse_val, "r2_val": r2_val}
            )

    # Select best by val RMSE
    best = min(results, key=lambda x: x["rmse_val"])
    print(f"\nBest model: {best['name']} (rmse_log={best['rmse_val']:.4f})")

    # Register best model
    client = MlflowClient()
    model_uri = f"runs:/{best['run_id']}/model"
    result = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)
    client.transition_model_version_stage(
        name=REGISTRY_NAME,
        version=result.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    print(f"Registered: {REGISTRY_NAME} v{result.version} → Staging ✅")

    # Save vectorizer feature count for reference
    print(f"\nDictVectorizer features: {len(dv.feature_names_)}")
    print("\n✅ Training complete!")
    return dv, best


if __name__ == "__main__":
    main()
