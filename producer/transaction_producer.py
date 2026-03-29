"""
----------------------
Generates synthetic bank transactions and publishes them to the
Kafka 'transactions' topic in real time.

"""

import json
import random
import time
import uuid
from datetime import datetime, timezone

from confluent_kafka import Producer
from dotenv import load_dotenv
from faker import Faker
from loguru import logger
import os

# 1. Config ────────────────────────────────────────────────────────────

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "transactions")
TRANSACTIONS_PER_SECOND = float(os.getenv("TRANSACTIONS_PER_SECOND", "10"))
FRAUD_RATE              = float(os.getenv("FRAUD_RATE", "0.02"))

fake = Faker()

# 2. Merchant ─────────────────────────────────────────────────

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "restaurant",
    "fuel", "pharmacy", "clothing", "entertainment"
]

MERCHANTS = [
    {"merchant_id": f"M{str(i).zfill(4)}", "category": random.choice(MERCHANT_CATEGORIES)}
    for i in range(200)
]

# 3. Customer ──────────────────────────────────────────────────

CUSTOMER_IDS = [f"C{str(i).zfill(5)}" for i in range(500)]

# 4. Amount Distributions ───────────────────────────────────────────────

def generate_amount(is_fraud: bool) -> float:
    if is_fraud:
        # Fraud: bimodal — either very small (testing card) or very large
        if random.random() < 0.3:
            return round(random.uniform(0.01, 5.00), 2)
        return round(random.uniform(200.00, 5000.00), 2)
    # Legitimate: log-normal distribution centred around $45
    return round(min(random.lognormvariate(3.8, 0.8), 2000.0), 2)


def generate_transaction(is_fraud: bool) -> dict:
    merchant    = random.choice(MERCHANTS)
    customer_id = random.choice(CUSTOMER_IDS)

    # Fraud signals in data. Higher rate of card-not-present, and higher rate of foreign transactions
    card_present      = random.random() > (0.7 if is_fraud else 0.2)
    location_country  = (
        fake.country_code() if (is_fraud and random.random() < 0.4)
        else "AU"
    )

    return {
        "transaction_id":    str(uuid.uuid4()),
        "customer_id":       customer_id,
        "merchant_id":       merchant["merchant_id"],
        "merchant_category": merchant["category"],
        "amount":            generate_amount(is_fraud),
        "currency":          "AUD",
        "transaction_time":  datetime.now(timezone.utc).isoformat(),
        "card_present":      card_present,
        "location_country":  location_country,
        "is_fraud":          is_fraud,
    }


# 5. Kafka delivery callback ────────────────────────────────────────────
# Called asynchronously by the producer for every message.

def delivery_callback(err, msg):
    if err:
        logger.error(f"Delivery failed | topic={msg.topic()} | error={err}")
    else:
        logger.debug(
            f"Delivered | topic={msg.topic()} "
            f"| partition={msg.partition()} | offset={msg.offset()}"
        )


# 6. Main loop ─────────────────────────────────────────────────────────
def main():
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

    logger.info(f"Producer started | topic={KAFKA_TOPIC} | rate={TRANSACTIONS_PER_SECOND}/s")

    interval    = 1.0 / TRANSACTIONS_PER_SECOND
    total_sent  = 0
    fraud_sent  = 0

    try:
        while True:
            is_fraud    = random.random() < FRAUD_RATE
            transaction = generate_transaction(is_fraud)

            producer.produce(
                topic     = KAFKA_TOPIC,
                key       = transaction["customer_id"],   # partition by customer
                value     = json.dumps(transaction),
                callback  = delivery_callback,
            )

            # poll() triggers delivery callbacks — call it regularly
            producer.poll(0)

            total_sent += 1
            if is_fraud:
                fraud_sent += 1

            # Log a summary every 100 transactions
            if total_sent % 100 == 0:
                logger.info(
                    f"Sent {total_sent} transactions | "
                    f"fraud={fraud_sent} ({fraud_sent/total_sent*100:.1f}%)"
                )

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Shutting down — flushing remaining messages...")
        producer.flush()
        logger.info(f"Done. Total sent: {total_sent} | Fraud: {fraud_sent}")


if __name__ == "__main__":
    main()