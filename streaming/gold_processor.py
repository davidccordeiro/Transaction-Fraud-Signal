"""
Reads the Silver Delta table and produces a Gold Delta table
enriched with ML features for fraud detection.

Design Decisions:
- velocity_1h: count of transactions by this customer in the last
  hour. High velocity is one of the strongest fraud signals.
- amount_zscore: how many standard deviations this transaction's
  amount is from the customer's historical mean. Computed using
  a 7-day rolling window.
- is_high_risk_country: binary flag for non-US transactions.
  Simple but effective — most domestic fraud is card-not-present.
- is_high_risk_hour: transactions between 01:00-05:00 UTC are
  statistically more likely to be fraudulent.
- is_high_risk_category: electronics and travel have the highest
  fraud rates by merchant category in retail banking.
"""

import os

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 1. Config ────────────────────────────────────────────────────────────

load_dotenv()

DELTA_SILVER_PATH = os.getenv("DELTA_SILVER_PATH", "/home/davidccordeiro/fraud-platform/data/delta/silver")
DELTA_GOLD_PATH   = os.getenv("DELTA_GOLD_PATH",   "/home/davidccordeiro/fraud-platform/data/delta/gold")
CHECKPOINT_GOLD   = "/home/davidccordeiro/fraud-platform/data/checkpoints/gold"
TRIGGER_INTERVAL  = "30 seconds"

# Feature engineering constants

HIGH_RISK_CATEGORIES = ["electronics", "travel"]
HIGH_RISK_HOURS      = list(range(1, 6))    # 01:00 - 05:00 UTC


# 2. Spark session ─────────────────────────────────────────────────────

def create_spark_session() -> SparkSession:
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        "--packages "
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
        "io.delta:delta-spark_2.12:3.1.0 "
        "pyspark-shell"
    )
    builder = (
        SparkSession.builder
        .appName("FraudPlatform-GoldFeatures")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


# ──3. Feature engineering ───────────────────────────────────────────────

def engineer_features(df: DataFrame) -> DataFrame:
    """
    Adds ML-ready feature columns to the Silver DataFrame.

    All features are computed using Spark window functions so they
    can be applied in a single pass without multiple aggregation jobs.
    """

    # ── Window: per customer, ordered by transaction time 

    customer_window = (
        Window
        .partitionBy("customer_id")
        .orderBy(F.col("transaction_time").cast("long"))
        .rangeBetween(-3600, 0)   # 1-hour lookback in seconds
    )

    # ── Window: per customer, all history 
    customer_full_window = (
        Window
        .partitionBy("customer_id")
        .orderBy(F.col("transaction_time").cast("long"))
        .rangeBetween(-604800, 0)  # 7-day lookback in seconds
    )

    df = (
        df
        # velocity_1h: how many transactions has this customer made in the last hour
        .withColumn(
            "velocity_1h",
            F.count("transaction_id").over(customer_window)
        )

        # amount_mean_7d / amount_stddev_7d: rolling stats for z-score
        .withColumn(
            "amount_mean_7d",
            F.mean("amount").over(customer_full_window)
        )
        .withColumn(
            "amount_stddev_7d",
            F.stddev("amount").over(customer_full_window)
        )

        # amount_zscore: standard deviations from customer's rolling mean
        # Coalesce to 0.0 when stddev is null (only one historical txn)
        .withColumn(
            "amount_zscore",
            F.when(
                F.col("amount_stddev_7d").isNull() | (F.col("amount_stddev_7d") == 0),
                F.lit(0.0)
            ).otherwise(
                (F.col("amount") - F.col("amount_mean_7d")) / F.col("amount_stddev_7d")
            )
        )

        # is_high_risk_hour: flag transactions in the 01:00-05:00 UTC window
        .withColumn(
            "is_high_risk_hour",
            F.hour(F.col("transaction_time")).isin(HIGH_RISK_HOURS)
        )

        # is_high_risk_country: any non-US transaction
        .withColumn(
            "is_high_risk_country",
            F.col("location_country") != "US"
        )

        # is_high_risk_category: electronics and travel
        .withColumn(
            "is_high_risk_category",
            F.col("merchant_category").isin(HIGH_RISK_CATEGORIES)
        )

        # card_not_present: inverse of card_present — cleaner ML feature name
        .withColumn(
            "card_not_present",
            ~F.col("card_present")
        )

        # gold_processed_at: SLA measurement timestamp
        .withColumn(
            "gold_processed_at",
            F.current_timestamp()
        )

        # Drop intermediate columns not needed by consumers
        .drop("amount_mean_7d", "amount_stddev_7d")
    )

    return df


# 4. Gold writer ───────────────────────────────────────────────────────

def write_gold_batch(batch_df: DataFrame, batch_id: int) -> None:
    if batch_df.isEmpty():
        return

    enriched_df  = engineer_features(batch_df)
    enriched_df  = enriched_df.dropDuplicates(["transaction_id"])
    record_count = enriched_df.count()

    spark = batch_df.sparkSession

    if not DeltaTable.isDeltaTable(spark, DELTA_GOLD_PATH):
        (
            enriched_df.write
            .format("delta")
            .mode("overwrite")
            .save(DELTA_GOLD_PATH)
        )
        logger.info(f"Batch {batch_id} | Gold table created | rows={record_count}")
    else:
        gold_table = DeltaTable.forPath(spark, DELTA_GOLD_PATH)
        (
            gold_table.alias("gold")
            .merge(
                enriched_df.alias("updates"),
                "gold.transaction_id = updates.transaction_id"
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
        logger.info(
            f"Batch {batch_id} | "
            f"rows={record_count} | "
            f"gold_path={DELTA_GOLD_PATH}"
        )


# 5. Main ──────────────────────────────────────────────────────────────

def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Starting Gold feature engineering stream...")
    logger.info(f"Reading from Silver : {DELTA_SILVER_PATH}")
    logger.info(f"Writing to Gold     : {DELTA_GOLD_PATH}")

    silver_stream_df = (
        spark.readStream
        .format("delta")
        .option("ignoreChanges", "true")
        .load(DELTA_SILVER_PATH)
    )

    query = (
        silver_stream_df
        .writeStream
        .foreachBatch(write_gold_batch)
        .option("checkpointLocation", CHECKPOINT_GOLD)
        .trigger(processingTime=TRIGGER_INTERVAL)
        .start()
    )

    logger.info(f"Gold stream running | trigger={TRIGGER_INTERVAL}")
    logger.info("Press Ctrl+C to stop.")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("Stopping Gold stream...")
        query.stop()
        spark.stop()
        logger.info("Gold stream stopped cleanly.")


if __name__ == "__main__":
    main()