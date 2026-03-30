"""
Loads the latest trained Isolation Forest from MLflow and scores
new transactions from the Gold Delta table.

This demonstrates the model serving → observability loop:
1. Load model from MLflow registry
2. Read unseen transactions from Gold
3. Score each transaction
4. Log prediction distribution back to MLflow
5. Flag high-risk transactions for downstream alerting

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

# 1. Config ────────────────────────────────────────────────────────────

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlflow/mlruns")
GOLD_PATH           = os.getenv("DELTA_GOLD_PATH", "/home/davidccordeiro/fraud-platform/data/delta/gold")
MODEL_NAME          = "fraud-isolation-forest"
MODEL_STAGE         = "latest"

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

# Percentile-based threshold — flag the top 2% most anomalous transactions as fraud

ANOMALY_PERCENTILE = 2


# 2. Load model ────────────────────────────────────────────────────────

def load_model():
    """
    Loads the latest registered Isolation Forest from MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info(f"Loading model from MLflow | uri={model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Model loaded successfully")
    return model


# 3. Load transactions ─────────────────────────────────────────────────

def load_transactions() -> pd.DataFrame:
    """
    Reads the Gold Delta table via DuckDB for scoring.
    """
    conn = duckdb.connect()
    df = conn.execute(f"""
        SELECT * FROM read_parquet(
            '{GOLD_PATH}/**/*.parquet',
            union_by_name=true
        )
    """).df()
    conn.close()

    # Cast booleans to int
    bool_cols = [
        "is_high_risk_hour",
        "is_high_risk_country",
        "is_high_risk_category",
        "card_not_present",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Add derived features if reading from Gold directly
    # (these exist in fct_fraud_features but not raw Gold)
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

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)
    return df


# 4. Score ─────────────────────────────────────────────────────────────

def score_transactions(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores each transaction and adds prediction columns.

    Uses a percentile-based threshold rather than a fixed value —
    flags the top ANOMALY_PERCENTILE% most anomalous transactions.
    This is more robust than a hardcoded threshold because it
    adapts to the score distribution of each dataset.

    Columns added:
    - anomaly_score     : raw Isolation Forest score (more negative = riskier)
    - is_predicted_fraud: True if score in bottom ANOMALY_PERCENTILE%
    - risk_tier         : high / medium / low based on score percentiles
    """
    X = df[FEATURE_COLUMNS].values

    df["anomaly_score"] = model.score_samples(X)

    # Dynamic threshold: bottom 2% of scores = predicted fraud
    threshold = np.percentile(df["anomaly_score"], ANOMALY_PERCENTILE)
    df["is_predicted_fraud"] = df["anomaly_score"] < threshold

    # Risk tiers based on score percentiles
    p10 = np.percentile(df["anomaly_score"], 10)
    p25 = np.percentile(df["anomaly_score"], 25)

    df["risk_tier"] = pd.cut(
        df["anomaly_score"],
        bins   = [-np.inf, p10, p25, np.inf],
        labels = ["high", "medium", "low"]
    )

    logger.info(f"Anomaly score threshold (p{ANOMALY_PERCENTILE}): {threshold:.4f}")

    return df


# 5. Observability ─────────────────────────────────────────────────────

def log_scoring_run(df: pd.DataFrame) -> None:
    """
    Logs prediction distribution metrics back to MLflow.
    This closes the observability loop — every scoring run
    is tracked so you can monitor model drift over time.
    """
    with mlflow.start_run(run_name="scoring-run"):
        n_total     = len(df)
        n_predicted = df["is_predicted_fraud"].sum()
        n_actual    = df["is_fraud"].sum() if "is_fraud" in df.columns else None

        mlflow.log_metrics({
            "n_scored":            n_total,
            "n_predicted_fraud":   int(n_predicted),
            "predicted_fraud_rate": n_predicted / n_total,
        })

        if n_actual is not None:
            mlflow.log_metrics({
                "n_actual_fraud":   int(n_actual),
                "actual_fraud_rate": n_actual / n_total,
            })

        logger.info(f"Scoring run logged | n={n_total} | predicted_fraud={n_predicted}")


# 6. Main ──────────────────────────────────────────────────────────────

def main():
    logger.info("Starting fraud scorer...")

    model = load_model()
    df    = load_transactions()

    logger.info(f"Loaded {len(df)} transactions for scoring")

    df = score_transactions(model, df)

    # Summary
    high_risk  = (df["risk_tier"] == "high").sum()
    med_risk   = (df["risk_tier"] == "medium").sum()
    low_risk   = (df["risk_tier"] == "low").sum()
    predicted  = df["is_predicted_fraud"].sum()

    logger.info(f"Scoring complete:")
    logger.info(f"  Total scored    : {len(df)}")
    logger.info(f"  Predicted fraud : {predicted} ({predicted/len(df):.2%})")
    logger.info(f"  High risk       : {high_risk}")
    logger.info(f"  Medium risk     : {med_risk}")
    logger.info(f"  Low risk        : {low_risk}")

    # Show sample high-risk transactions
    high_risk_df = df[df["risk_tier"] == "high"][[
        "transaction_id", "customer_id", "amount",
        "anomaly_score", "risk_tier", "is_fraud"
    ]].head(10)

    logger.info(f"\nSample high-risk transactions:\n{high_risk_df.to_string()}")

    log_scoring_run(df)
    logger.info("Done.")


if __name__ == "__main__":
    main()