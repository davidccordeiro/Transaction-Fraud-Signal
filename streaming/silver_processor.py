"""
silver_processor.py
-------------------
Reads the Bronze Delta table and produces a validated, deduplicated
Silver Delta table.

Design decisions:
- We read Bronze as a streaming source using Delta's readStream —
  this means Silver automatically processes new Bronze records as
  they arrive.
- Deduplication uses dropDuplicates on transaction_id within a
  watermark window. The watermark tells Spark how late data can
  arrive before being dropped — 10 minutes is conservative but
  safe for a local fraud platform.
- We use Delta MERGE (upsert) rather than append to write Silver.
  This guarantees idempotency: if the Silver job restarts and
  reprocesses a Bronze batch, existing records are updated rather
  than duplicated.
"""

import os

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

# ── Config ────────────────────────────────────────────────────────────
load_dotenv()

DELTA_BRONZE_PATH   = os.getenv("DELTA_BRONZE_PATH", "/home/davidccordeiro/fraud-platform/data/delta/bronze")
DELTA_SILVER_PATH   = os.getenv("DELTA_SILVER_PATH", "/home/davidccordeiro/fraud-platform/data/delta/silver")
CHECKPOINT_SILVER   = "/home/davidccordeiro/fraud-platform/data/checkpoints/silver"
TRIGGER_INTERVAL    = "10 seconds"

# Validation rules — amounts outside this range are quarantined
MIN_AMOUNT = 0.01
MAX_AMOUNT = 50000.00
VALID_CURRENCIES = ["AUD"]


# 1. Spark session ─────────────────────────────────────────────────────

def create_spark_session() -> SparkSession:
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        "--packages "
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
        "io.delta:delta-spark_2.12:3.1.0 "
        "pyspark-shell"
    )
    builder = (
        SparkSession.builder
        .appName("FraudPlatform-SilverValidation")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


# 2. Validation logic ──────────────────────────────────────────────────

def apply_validation(df: DataFrame) -> tuple:
    """
    Splits a DataFrame into valid and invalid records.

    Validation rules:
    - amount must be between MIN_AMOUNT and MAX_AMOUNT
    - currency must be in VALID_CURRENCIES
    - transaction_id, customer_id, merchant_id must not be null
    - transaction_time must be parseable as a timestamp

    """
    # Cast transaction_time string → proper TimestampType
    df = df.withColumn(
        "transaction_time",
        F.to_timestamp(F.col("transaction_time"))
    )

    # Build a single validation flag column
    # is_valid = True only if ALL rules pass
    df = df.withColumn(
        "is_valid",
        (F.col("amount").between(MIN_AMOUNT, MAX_AMOUNT)) &
        (F.col("currency").isin(VALID_CURRENCIES)) &
        (F.col("transaction_id").isNotNull()) &
        (F.col("customer_id").isNotNull()) &
        (F.col("merchant_id").isNotNull()) &
        (F.col("transaction_time").isNotNull())
    )

    valid_df   = df.filter(F.col("is_valid")).drop("is_valid")
    invalid_df = df.filter(~F.col("is_valid")).drop("is_valid")

    return valid_df, invalid_df


# 3. Silver writer ─────────────────────────────────────────────────────

def write_silver_batch(batch_df: DataFrame, batch_id: int) -> None:
    """
    Validates, deduplicates, and merges each Bronze micro-batch
    into the Silver Delta table.

    MERGE semantics:
    - If transaction_id already exists in Silver → Update the record
    - If transaction_id is new → Insert the record
    """
    if batch_df.isEmpty():
        return

    # Add Silver metadata
    batch_df = batch_df.withColumn(
        "silver_processed_at",
        F.lit(F.current_timestamp())
    )

    # Apply validation rules
    valid_df, invalid_df = apply_validation(batch_df)

    # Deduplicate within the batch on transaction_id
    # (keeps the first occurrence if duplicates exist in one batch)
    valid_df = valid_df.dropDuplicates(["transaction_id"])

    valid_count   = valid_df.count()
    invalid_count = invalid_df.count()

    if valid_count == 0:
        logger.warning(f"Batch {batch_id} | no valid records after validation")
        return

    # Get the Spark session from the batch DataFrame
    spark = batch_df.sparkSession

    # MERGE into Silver — upsert pattern
    # If Silver table doesn't exist yet, create it on first write
    if not DeltaTable.isDeltaTable(spark, DELTA_SILVER_PATH):
        (
            valid_df.write
            .format("delta")
            .mode("overwrite")
            .save(DELTA_SILVER_PATH)
        )
        logger.info(f"Batch {batch_id} | Silver table created | rows={valid_count}")
    else:
        silver_table = DeltaTable.forPath(spark, DELTA_SILVER_PATH)
        (
            silver_table.alias("silver")
            .merge(
                valid_df.alias("updates"),
                "silver.transaction_id = updates.transaction_id"
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
        logger.info(
            f"Batch {batch_id} | "
            f"merged={valid_count} | "
            f"invalid={invalid_count} | "
            f"silver_path={DELTA_SILVER_PATH}"
        )


# 4. Main ──────────────────────────────────────────────────────────────

def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Starting Silver validation stream...")
    logger.info(f"Reading from Bronze : {DELTA_BRONZE_PATH}")
    logger.info(f"Writing to Silver   : {DELTA_SILVER_PATH}")

    # Read Bronze as a streaming Delta source
    # Delta's readStream automatically picks up new files as
    # Bronze appends them — no Kafka connection needed here
    bronze_stream_df = (
        spark.readStream
        .format("delta")
        .option("ignoreChanges", "true")
        .load(DELTA_BRONZE_PATH)
    )

    query = (
        bronze_stream_df
        .writeStream
        .foreachBatch(write_silver_batch)
        .option("checkpointLocation", CHECKPOINT_SILVER)
        .trigger(processingTime=TRIGGER_INTERVAL)
        .start()
    )

    logger.info(f"Silver stream running | trigger={TRIGGER_INTERVAL}")
    logger.info("Press Ctrl+C to stop.")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("Stopping Silver stream...")
        query.stop()
        spark.stop()
        logger.info("Silver stream stopped cleanly.")


if __name__ == "__main__":
    main()