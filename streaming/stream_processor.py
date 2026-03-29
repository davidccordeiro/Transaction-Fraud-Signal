"""
-------------------
PySpark Structured Streaming job that reads raw transactions
from Kafka and writes them to the Bronze Delta Lake layer.

Design decisions:
- We use foreachBatch rather than a direct Delta sink so we can
  add per-batch observability (row counts, fraud rates) without
  a separate monitoring job.
- Schema enforcement happens here, not in Silver, so malformed
  records never pollute the lakehouse. They land in a quarantine
  path for investigation.
- checkpointLocation is mandatory for Structured Streaming —
  it tracks exactly which Kafka offsets have been committed so
  the job can resume without data loss or duplication after a
  restart.

"""

import os
from datetime import datetime, timezone

from delta import configure_spark_with_delta_pip
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    FloatType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
# 1. Config ────────────────────────────────────────────────────────────

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "transactions")
DELTA_BRONZE_PATH       = os.getenv("DELTA_BRONZE_PATH", "/home/davidccordeiro/fraud-platform/data/delta/bronze")
CHECKPOINT_BRONZE       = "/home/davidccordeiro/fraud-platform/data/checkpoints/bronze"
QUARANTINE_PATH         = "/home/davidccordeiro/fraud-platform/data/quarantine/bronze"
TRIGGER_INTERVAL        = "2 seconds"

# 2. Transaction schema ────────────────────────────────────────────────

TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id",    StringType(),    nullable=False),
    StructField("customer_id",       StringType(),    nullable=False),
    StructField("merchant_id",       StringType(),    nullable=False),
    StructField("merchant_category", StringType(),    nullable=True),
    StructField("amount",            FloatType(),     nullable=False),
    StructField("currency",          StringType(),    nullable=True),
    StructField("transaction_time",  StringType(),    nullable=False),
    StructField("card_present",      BooleanType(),   nullable=True),
    StructField("location_country",  StringType(),    nullable=True),
    StructField("is_fraud",          BooleanType(),   nullable=True),
])


# 3. Spark session ─────────────────────────────────────────────────────

def create_spark_session() -> SparkSession:
    """
    Builds a SparkSession with Delta Lake and Kafka support.

    Both the Kafka and Delta JARs are passed via PYSPARK_SUBMIT_ARGS
    so they are resolved together before SparkContext initialises.
    configure_spark_with_delta_pip then adds the Delta pip package
    on top without conflicting with the already-loaded JARs.
    """
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        "--packages "
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
        "io.delta:delta-spark_2.12:3.1.0 "
        "pyspark-shell"
    )

    builder = (
        SparkSession.builder
        .appName("FraudPlatform-BronzeIngestion")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.extraJavaOptions", "-Dlog4j.logLevel=WARN")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()

# 4. Bronze writer ─────────────────────────────────────────────────────

def write_bronze_batch(batch_df: DataFrame, batch_id: int) -> None:
    """
    Called by Structured Streaming for each micro-batch.

    Splits the batch into valid and quarantined records, and also logs per-batch metrics for observability.
    """
    if batch_df.isEmpty():
        logger.debug(f"Batch {batch_id} | empty batch, skipping")
        return

    # Parse JSON from Kafka value (raw bytes → structured columns)

    parsed_df = batch_df.select(
        F.from_json(
            F.col("value").cast("string"),
            TRANSACTION_SCHEMA
        ).alias("data"),
        F.col("timestamp").alias("kafka_timestamp"),
        F.col("partition"),
        F.col("offset"),
    ).select(
        "data.*",
        "kafka_timestamp",
        "partition",
        "offset",
    )

    # Add ingestion metadata columns

    enriched_df = parsed_df.withColumn(
        "ingested_at", F.lit(datetime.now(timezone.utc).isoformat())
    ).withColumn(
        "batch_id", F.lit(batch_id)
    )

    # Split valid vs quarantine
    # A record is invalid if critical nullable=False fields are null

    valid_df = enriched_df.filter(
        F.col("transaction_id").isNotNull() &
        F.col("customer_id").isNotNull() &
        F.col("amount").isNotNull()
    )

    quarantine_df = enriched_df.filter(
        F.col("transaction_id").isNull() |
        F.col("customer_id").isNull() |
        F.col("amount").isNull()
    )

    valid_count     = valid_df.count()
    quarantine_count = quarantine_df.count()

    # Write valid records to Bronze Delta table
    if valid_count > 0:
        (
            valid_df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")   # allows schema evolution
            .save(DELTA_BRONZE_PATH)
        )

    # Write quarantined records as JSON for investigation

    if quarantine_count > 0:
        (
            quarantine_df.write
            .format("json")
            .mode("append")
            .save(QUARANTINE_PATH)
        )
        logger.warning(f"Batch {batch_id} | quarantined {quarantine_count} malformed records")

    logger.info(
        f"Batch {batch_id} | "
        f"valid={valid_count} | "
        f"quarantined={quarantine_count} | "
        f"bronze_path={DELTA_BRONZE_PATH}"
    )


# 5. Main ──────────────────────────────────────────────────────────────

def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Starting Bronze ingestion stream...")
    logger.info(f"Reading from Kafka topic: {KAFKA_TOPIC}")
    logger.info(f"Writing to Bronze path  : {DELTA_BRONZE_PATH}")

    # Read from Kafka as an unbounded stream

    kafka_stream_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 1000)
        .load()
    )

    # Start the streaming query
    query = (
        kafka_stream_df
        .writeStream
        .foreachBatch(write_bronze_batch)
        .option("checkpointLocation", CHECKPOINT_BRONZE)
        .trigger(processingTime=TRIGGER_INTERVAL)
        .start()
    )

    logger.info(f"Bronze stream running | trigger={TRIGGER_INTERVAL}")
    logger.info("Press Ctrl+C to stop.")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("Stopping Bronze stream...")
        query.stop()
        spark.stop()
        logger.info("Bronze stream stopped cleanly.")


if __name__ == "__main__":
    main()