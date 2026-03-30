"""
Trains an Isolation Forest fraud detector on the Gold feature
table and logs the experiment to MLflow.

Design decisions:
- Isolation Forest over logistic regression because we treat
  fraud detection as anomaly detection, not classification.
- We read from the dbt mart (fct_fraud_features) via DuckDB
  rather than directly from Delta parquet — this validates
  the full pipeline end-to-end.

"""

import os
import warnings
warnings.filterwarnings("ignore")

import duckdb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────
load_dotenv()

MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "/home/davidccordeiro/fraud-platform/mlflow/mlruns")
MLFLOW_EXPERIMENT     = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
GOLD_PATH             = os.getenv("DELTA_GOLD_PATH", "/home/davidccordeiro/fraud-platform/data/delta/gold")
DBT_DB_PATH           = "/home/davidccordeiro/fraud-platform/fraud_dbt/dev.duckdb"

# Model Hyperparameters

CONTAMINATION     = 0.02    # Expected fraud rate — matches our producer
N_ESTIMATORS      = 100
MAX_SAMPLES       = "auto"
RANDOM_STATE      = 42

# Features the model trains on. Are all available in fct_fraud_features

FEATURE_COLUMNS = [
    "amount",
    "amount_zscore",
    "velocity_1h",
    "is_high_risk_hour",
    "is_high_risk_country",
    "is_high_risk_category",
    "card_not_present",
    "fraud_risk_score",
    "transaction_hour",
    "transaction_dow",
]


# 1. Data loading ──────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """
    Reads the fct_fraud_features dbt mart via DuckDB.
    Falls back to reading Gold parquet directly if the dbt
    database hasn't been built yet.
    """
    try:
        conn = duckdb.connect(DBT_DB_PATH)
        df = conn.execute(
            "SELECT * FROM fraud_dbt_marts.fct_fraud_features"
        ).df()
        conn.close()
        logger.info(f"Loaded {len(df)} rows from dbt mart")
        return df
    except Exception as e:
        logger.warning(f"dbt mart unavailable ({e}), falling back to Gold parquet")
        conn = duckdb.connect()
        df = conn.execute(f"""
            SELECT * FROM read_parquet(
                '{GOLD_PATH}/**/*.parquet',
                union_by_name=true
            )
        """).df()
        conn.close()
        logger.info(f"Loaded {len(df)} rows from Gold parquet directly")
        return df


# 2. Feature preparation ───────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepares the feature matrix X and label vector y.

    Boolean columns are cast to int (True→1, False→0).
    Derived features are computed here if reading from Gold
    parquet directly rather than the dbt mart.
    Returns (X, y, feature_names).
    """
    # Cast boolean features to int for sklearn
    bool_cols = [
        "is_high_risk_hour",
        "is_high_risk_country",
        "is_high_risk_category",
        "card_not_present",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Compute derived features if not already present
    # (these come from dbt mart — recompute when falling back to Gold)
    if "transaction_hour" not in df.columns:
        df["transaction_time"] = pd.to_datetime(df["transaction_time"])
        df["transaction_hour"] = df["transaction_time"].dt.hour
        df["transaction_dow"]  = df["transaction_time"].dt.dayofweek

    if "fraud_risk_score" not in df.columns:
        df["fraud_risk_score"] = (
            df["is_high_risk_hour"].astype(int) +
            df["is_high_risk_country"].astype(int) +
            df["is_high_risk_category"].astype(int) +
            df["card_not_present"].astype(int) +
            df["velocity_1h"].apply(lambda v: 2 if v >= 3 else 0)
        )

    # Fill any nulls with 0
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    X = df[FEATURE_COLUMNS].values
    y = df["is_fraud"].astype(int).values

    return X, y

# 3. Evaluation ────────────────────────────────────────────────────────

def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluates the Isolation Forest against ground truth labels.

    Isolation Forest predicts -1 (anomaly) or 1 (normal).
    We convert to 1 (fraud) / 0 (normal) to match our label.

    Ground truth labels are purely used here for observability purposes. The goal is to track metrics in MLflow, not accuracy.
    """
    raw_preds = model.predict(X)
    # Convert: -1 (anomaly) → 1 (fraud), 1 (normal) → 0 (legit)
    y_pred = np.where(raw_preds == -1, 1, 0)

    # Anomaly scores — more negative = more anomalous
    scores = model.score_samples(X)
    # Invert for ROC AUC (higher score = more likely fraud)
    scores_inverted = -scores

    metrics = {
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall":    recall_score(y, y_pred, zero_division=0),
        "f1":        f1_score(y, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y, scores_inverted),
        "n_samples": len(y),
        "n_fraud":   int(y.sum()),
        "n_legit":   int((y == 0).sum()),
    }

    logger.info("Model evaluation:")
    logger.info(f"  Precision : {metrics['precision']:.3f}")
    logger.info(f"  Recall    : {metrics['recall']:.3f}")
    logger.info(f"  F1        : {metrics['f1']:.3f}")
    logger.info(f"  ROC AUC   : {metrics['roc_auc']:.3f}")
    logger.info(f"  Samples   : {metrics['n_samples']} "
                f"(fraud={metrics['n_fraud']}, legit={metrics['n_legit']})")

    return metrics


# 4. Main ──────────────────────────────────────────────────────────────

def main():
    # Point MLflow at our local tracking directory
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    logger.info("Loading features...")
    df = load_features()

    if len(df) < 50:
        logger.error(
            f"Only {len(df)} rows available. Run the pipeline for longer "
            "to generate enough training data (minimum 50 rows)."
        )
        return

    X, y = prepare_features(df)
    logger.info(f"Feature matrix: {X.shape} | Fraud rate: {y.mean():.2%}")

    with mlflow.start_run() as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log hyperparameters
        mlflow.log_params({
            "model_type":    "IsolationForest",
            "contamination": CONTAMINATION,
            "n_estimators":  N_ESTIMATORS,
            "max_samples":   MAX_SAMPLES,
            "random_state":  RANDOM_STATE,
            "n_features":    len(FEATURE_COLUMNS),
            "features":      ",".join(FEATURE_COLUMNS),
        })

        # Train
        logger.info("Training Isolation Forest...")
        model = IsolationForest(
            contamination = CONTAMINATION,
            n_estimators  = N_ESTIMATORS,
            max_samples   = MAX_SAMPLES,
            random_state  = RANDOM_STATE,
            n_jobs        = -1,
        )
        model.fit(X)

        # Evaluate
        metrics = evaluate_model(model, X, y)
        mlflow.log_metrics(metrics)

        # Log the model artifact
        mlflow.sklearn.log_model(
            sk_model        = model,
            artifact_path   = "isolation_forest",
            registered_model_name = "fraud-isolation-forest",
        )

        logger.info(f"Model logged to MLflow | run_id={run.info.run_id}")
        logger.info(f"Tracking URI: {MLFLOW_TRACKING_URI}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()