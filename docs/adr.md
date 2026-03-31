# ADR-001: Real-Time Fraud Signal Platform — Architecture Decisions


## Context

A retail bank processes millions of transactions daily but detects
fraud hours after the fact using nightly batch jobs. By the time
fraud is detected, the financial damage is done and the customer
experience is already broken.

The goal is to build a streaming ingestion and feature engineering
platform that surfaces anomalous transaction signals within seconds
of occurrence, enabling a downstream ML model to score each
transaction in near real-time.

This ADR documents the key architectural decisions made during
the design of this platform and the trade-offs considered.

---

## Decision 1: Streaming over Micro-Batch

### Decision
Use true event-driven streaming (sub-second record processing)
rather than micro-batch processing (small batches on a fixed
interval, e.g. every 5 minutes).

### Rationale
Fraud has a time dimension that batch processing fundamentally
cannot address. The longer a fraudulent transaction goes undetected,
the more damage occurs:

- Card testing attacks happen in bursts of seconds — a 5-minute
  micro-batch window misses the velocity signal entirely
- Customers expect real-time decline decisions at point of sale
- Regulatory frameworks (PSD2, Reg E) increasingly require
  demonstrable real-time monitoring

Streaming with a 2-second trigger interval gives us a Bronze
SLA of < 5 seconds end-to-end — sufficient for real-time
fraud alerting without the operational complexity of true
sub-millisecond streaming.

### Trade-offs
| | Streaming | Micro-Batch |
|---|---|---|
| Latency | < 5 seconds | 1-15 minutes |
| Complexity | Higher | Lower |
| Cost | Higher (always-on) | Lower (runs periodically) |
| Fraud signal quality | High (velocity preserved) | Low (velocity lost) |
| Operational burden | Requires 24/7 monitoring | Simpler to operate |

### Rejected alternative
Spark micro-batch on a 5-minute schedule. Rejected because
velocity-based fraud signals (e.g. 10 transactions in 60 seconds)
are completely invisible at 5-minute granularity — the defining
feature of card testing attacks.

---

## Decision 2: Apache Kafka over AWS Kinesis

### Decision
Use Apache Kafka (self-hosted via Docker) as the message queue
rather than AWS Kinesis Data Streams.

### Rationale

**Portability.** Kafka runs identically on-premises, in Docker,
on Azure (Confluent Cloud or HDInsight), and on AWS (MSK). The
platform is not tied to any cloud provider. This matters for a
retail bank that may have regulatory requirements around data
residency or vendor lock-in policies.

**Partition semantics.** Kafka's partitioning model allows us to
key messages by `customer_id`, guaranteeing that all transactions
for a given customer land on the same partition and are processed
in order. This is critical for velocity calculations — out-of-order
processing would corrupt the velocity feature. Kinesis uses shards
with a similar keying model but with less flexibility in partition
management.

**Ecosystem.** Kafka has first-class connectors to every major
data system — Delta Lake, Spark, Flink, dbt, Debezium for CDC.
Kinesis is deeply integrated with the AWS ecosystem but requires
additional tooling (Kinesis Firehose, Lambda) to achieve the same
connectivity.

**Cost at scale.** Kafka on Azure (Confluent Cloud) costs
approximately $0.10/GB ingested. Kinesis costs $0.015 per shard-
hour plus $0.08/GB — at high throughput (>1TB/day) Kafka is
significantly cheaper.

### Trade-offs
| | Kafka | Kinesis |
|---|---|---|
| Operational overhead | High (self-managed) | Low (fully managed) |
| Cloud portability | High | AWS only |
| Partition flexibility | High | Medium |
| Cost at scale | Lower | Higher |
| Time to production | Slower | Faster |

### Rejected alternative
AWS Kinesis Data Streams. Rejected primarily on portability
grounds — the bank's cloud strategy is Azure-first, and Kinesis
would require a full re-architecture if the platform needed to
run on-premises or on Azure.

---

## Decision 3: Delta Lake over Apache Iceberg

### Decision
Use Delta Lake OSS as the lakehouse table format rather than
Apache Iceberg.

### Rationale

**PySpark integration.** Delta Lake was built by Databricks
specifically for Spark and has the deepest, most battle-tested
integration with PySpark Structured Streaming. Iceberg's Spark
integration is good but lags Delta on streaming-specific features
like schema enforcement on ingest and MERGE-based upserts in
streaming contexts.

**MERGE semantics.** Our Silver layer uses Delta MERGE to achieve
idempotent upserts — if the streaming job restarts and reprocesses
a Bronze batch, existing Silver records are updated rather than
duplicated. Delta's MERGE implementation is more mature and
performant than Iceberg's for our write patterns.

**Schema evolution.** Delta Lake's `mergeSchema` option allows
the Bronze table to automatically evolve when the producer adds
new fields — without requiring a schema migration job. This is
important for a fraud platform where new transaction signals are
regularly added by the data science team.

**Azure ecosystem.** Azure Databricks (the likely production
platform for this workload) has native Delta Lake support built
in. Iceberg on Azure requires additional configuration and is
not as deeply integrated with Azure Synapse or Azure Databricks.

### Trade-offs
| | Delta Lake | Iceberg |
|---|---|---|
| Spark integration | Excellent | Good |
| Multi-engine support | Good (improving) | Excellent |
| MERGE performance | High | Medium |
| Schema evolution | Automatic | Manual |
| Vendor neutrality | Databricks-led | Apache-led |
| Azure native support | First-class | Requires config |

### Rejected alternative
Apache Iceberg. Rejected because its primary advantage —
multi-engine support (Spark, Flink, Trino, Presto) — is not
a requirement for this platform. We run a single compute engine
(Spark) against a single cloud (Azure). Delta Lake's superior
Spark and Azure integration outweighs Iceberg's portability
advantage for this specific use case.

Note: if the bank required Trino or Presto for ad-hoc querying
by analysts alongside Spark streaming jobs, Iceberg would be
the correct choice. The decision is context-dependent.

---

## Decision 4: dbt Core with DuckDB over a Cloud Warehouse

### Decision
Use dbt Core with the DuckDB adapter to build feature models
on top of the Gold Delta Lake layer, rather than materialising
data into a cloud data warehouse (Snowflake, Azure Synapse).

### Rationale

**Zero additional infrastructure.** DuckDB is an embedded
analytical database that reads Delta Lake parquet files directly
from disk — no warehouse cluster to provision, no data to copy,
no egress costs. For a feature engineering layer that sits
between the lakehouse and the ML model, this is architecturally
elegant.

**SQL-native feature engineering.** dbt brings software
engineering practices to SQL — version control, testing, 
documentation, and dependency management. The `fct_fraud_features`
mart is tested on every run against the actual data, catching
data quality regressions before they reach the ML model.

**Production path.** In production, swapping DuckDB for
`dbt-synapse` or `dbt-databricks` is a single line change in
`profiles.yml`. The SQL models are portable across adapters.

### Trade-offs
| | dbt + DuckDB | Cloud Warehouse |
|---|---|---|
| Infrastructure cost | Zero | $200-2000/month |
| Query performance | Good (single node) | Excellent (distributed) |
| Concurrency | Low | High |
| Production readiness | Medium | High |
| Setup complexity | Low | High |

---

## Decision 5: Isolation Forest over Logistic Regression

### Decision
Use Isolation Forest (unsupervised anomaly detection) rather
than Logistic Regression (supervised classification) as the
fraud scorer.

### Rationale

**Label quality.** Fraud labels in retail banking are noisy,
delayed (chargebacks take 30-90 days), and heavily imbalanced
(~0.1-2% fraud rate). Logistic Regression trained on noisy
labels learns the noise. Isolation Forest learns what normal
looks like and flags deviations — more robust to label quality
issues.

**Cold start.** A new fraud platform has no labelled history.
Isolation Forest can be trained on unlabelled transaction data
from day one. Logistic Regression requires labelled examples
of both classes before it can produce meaningful predictions.

**Interpretability.** The `anomaly_score` output from Isolation
Forest is intuitive — more negative means more anomalous. The
`fraud_risk_score` rules-based pre-scorer provides an additional
interpretability layer that fraud analysts can reason about
without ML expertise.

### Trade-offs
| | Isolation Forest | Logistic Regression |
|---|---|---|
| Labelled data required | No | Yes |
| Class imbalance handling | Excellent | Requires resampling |
| Interpretability | Medium | High |
| Accuracy (with labels) | Medium | High |
| Cold start capability | Yes | No |

---

## SLA Summary

| Layer | Trigger | Target Latency | Retention |
|-------|---------|----------------|-----------|
| Bronze | 2 seconds | < 5 seconds | 7 days |
| Silver | 10 seconds | < 30 seconds | 30 days |
| Gold | 30 seconds | < 2 minutes | 90 days |
| ML Score | On demand | < 5 seconds | N/A |

---

## Risks and Mitigations

**Single Kafka broker.** A single broker is a single point of
failure. Mitigation: in production, run a 3-broker cluster with
replication factor 3. The `docker-compose.yml` is designed to
scale to multi-broker with minimal changes.

**Local Spark mode.** `local[2]` mode uses the driver as both
master and executor. Mitigation: in production, deploy on Azure
Databricks with auto-scaling clusters.

**No schema registry.** Schema evolution is handled by Delta
Lake's `mergeSchema` option rather than a Confluent Schema
Registry. Mitigation: acceptable for a single producer. With
multiple producers, add a Confluent Schema Registry container
to enforce Avro schemas at the Kafka level.