# Real-Time Fraud Signal Platform

A streaming data engineering portfolio project demonstrating near-real-time
fraud detection for financial transactions using a modern open-source stack.

## Architecture
```
Kafka (Synthetic Txns)
  → PySpark Structured Streaming
    → Delta Lake (Bronze → Silver → Gold)
      → dbt Core (Feature Models)
        → MLflow (Fraud Scorer + Observability)
```

Orchestrated by Prefect. Data quality enforced by Great Expectations.
Infrastructure defined with Terraform (local). All tooling is free and open-source.

## Medallion SLA Targets

| Layer  | Description                        | Latency Target |
|--------|------------------------------------|----------------|
| Bronze | Raw ingestion, no transformation   | < 5 seconds    |
| Silver | Validated, deduplicated, typed     | < 30 seconds   |
| Gold   | Business-ready, ML feature-ready   | < 2 minutes    |

## Tech Stack

| Tool                      | Role                              |
|---------------------------|-----------------------------------|
| Apache Kafka              | Message queue / event stream      |
| PySpark Structured Stream | Stream processing + features      |
| Delta Lake (OSS)          | Lakehouse storage (medallion)     |
| dbt Core                  | SQL feature models                |
| MLflow                    | Model tracking + serving          |
| Prefect                   | Pipeline orchestration            |
| Great Expectations        | Data quality contracts            |
| Terraform                 | Local infrastructure-as-code      |

## Prerequisites

- Windows 10/11 with WSL2 (Ubuntu 22.04)
- Docker Desktop 4.x+ with WSL2 backend enabled
- Miniconda (installed inside Ubuntu 22.04)

## Quick Start

### 1. Clone and enter the project
```bash
cd "/mnt/d/File Directory/Transaction-Fraud-Signal"
```

### 2. Start the infrastructure
```bash
docker compose up -d
```

Starts Zookeeper, Kafka, and Kafka UI. Wait ~30 seconds for all
health checks to pass.

### 3. Verify Kafka is running

Visit http://localhost:8080 in your browser.
You should see the `local-fraud-platform` cluster with 0 topics (none created yet).

### 4. Create the conda environment
```bash
conda create -n fraud-platform python=3.10 -y
conda activate fraud-platform
pip install -r requirements.txt
```

### 5. Stop the infrastructure
```bash
docker compose down
```

To also delete all stored data and volumes:
```bash
docker compose down -v
```

## Project Structure
```
Transaction-Fraud-Signal/
├── docker-compose.yml          # Kafka + Zookeeper + Kafka UI
├── .env                        # Environment config (never commit secrets)
├── .gitignore                  # Excludes .env, data/, mlruns/
├── README.md                   # This file
├── requirements.txt            # Python dependencies (Phase 2)
├── data/
│   └── delta/
│       ├── bronze/             # Raw ingested transactions
│       ├── silver/             # Validated transactions
│       └── gold/               # ML-ready feature table
├── producer/
│   └── transaction_producer.py # Synthetic Kafka producer (Phase 3)
├── streaming/
│   ├── stream_processor.py     # Bronze ingestion (Phase 4)
│   ├── silver_processor.py     # Silver validation (Phase 5)
│   └── gold_processor.py       # Gold feature engineering (Phase 5)
├── dbt/                        # dbt project (Phase 6)
├── ml/
│   ├── train.py                # Model training (Phase 7)
│   └── score.py                # Real-time scoring (Phase 7)
├── mlflow/                     # MLflow tracking store
├── docs/
│   ├── ADR.md                  # Architecture Decision Record (Phase 8)
│   └── cost_estimate.md        # Azure production cost model (Phase 8)
└── terraform/                  # Local infra definitions (Phase 8)
```

## Architecture Decision Record

See [`docs/ADR.md`](docs/ADR.md) for the full rationale behind every major
technology choice — including why Kafka over Kinesis, Delta Lake over Iceberg,
and streaming over micro-batch for this use case.

## Cost Estimate

See [`docs/cost_estimate.md`](docs/cost_estimate.md) for a production Azure
cost breakdown. The local setup costs nothing to run.

## Git Setup

Initialise the repo and protect secrets:
```bash
git init
git add .
git commit -m "phase 1: environment setup — kafka, docker, readme"
```

