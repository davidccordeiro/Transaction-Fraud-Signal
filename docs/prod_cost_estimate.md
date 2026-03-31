# Production Cost Estimate — Azure Deployment

**Scenario:** Retail bank processing 1,000 transactions/second  
**Region:** Australia East (Sydney)  
**Environment:** Production, 24/7 operation  

---

## Architecture Mapping (Local → Azure)

| Local Component | Azure Equivalent |
|----------------|-----------------|
| Kafka (Docker) | Confluent Cloud on Azure |
| PySpark Streaming | Azure Databricks (Streaming) |
| Delta Lake (local) | Delta Lake on Azure Data Lake Storage Gen2 |
| dbt Core | dbt Cloud (or self-hosted on AKS) |
| MLflow | Azure Databricks MLflow (managed) |
| Prefect | Prefect Cloud (or self-hosted) |

---

## Monthly Cost Breakdown

### 1. Confluent Cloud (Kafka) — ~$1,200/month
- Dedicated cluster: 2 CKUs (Confluent Kafka Units)
- 1,000 txn/sec × 500 bytes avg = ~43GB/day ingested
- Estimated: $800/month cluster + $400/month networking
- Replication factor 3 across 3 availability zones

### 2. Azure Databricks (Streaming + ML) — ~$3,500/month
- Bronze/Silver/Gold streaming: 2× Standard_DS3_v2
  (4 cores, 14GB RAM) running 24/7
  = 2 × $0.19/hour × 730 hours = ~$278/month compute
- Databricks Units (DBU): ~8 DBU/hour streaming
  = 8 × 730 × $0.40/DBU = ~$2,336/month
- ML training cluster (4 hours/week):
  = ~$150/month
- Total Databricks: ~$2,764/month + workspace overhead ~$3,500

### 3. Azure Data Lake Storage Gen2 (Delta Lake) — ~$200/month
- Bronze: 7 days × 43GB/day = ~300GB = $6/month
- Silver: 30 days × 40GB/day = ~1.2TB = $23/month
- Gold: 90 days × 38GB/day = ~3.4TB = $66/month
- Transactions (reads/writes): ~$50/month
- Snapshots and Delta logs: ~$55/month

### 4. Azure Kubernetes Service (dbt + Prefect) — ~$300/month
- 2× Standard_B2s nodes (2 cores, 4GB RAM)
- dbt runs every 15 minutes: lightweight, fits on small nodes
- Prefect agent co-located on same nodes

### 5. Azure Monitor + Log Analytics — ~$150/month
- Log ingestion: ~5GB/day pipeline logs
- Retention: 30 days
- Alerts and dashboards: included

### 6. Networking — ~$150/month
- Egress between services within region: minimal
- Egress to on-premises (if applicable): variable
- Private endpoints for security: included in compute

---

## Total Monthly Estimate

| Component | Monthly Cost |
|-----------|-------------|
| Confluent Cloud (Kafka) | $1,200 |
| Azure Databricks | $3,500 |
| Azure Data Lake Gen2 | $200 |
| AKS (dbt + Prefect) | $300 |
| Azure Monitor | $150 |
| Networking | $150 |
| **Total** | **~$5,500/month** |
| **Annual** | **~$66,000/year** |

---

## Cost Optimisation Levers

**Databricks spot instances.** Use spot (preemptible) instances
for the Gold layer and ML training — these are fault-tolerant
workloads that can restart on interruption. Saves ~40% on
Databricks compute = ~$1,000/month saving.

**Delta Lake VACUUM.** Run `VACUUM` weekly to remove old parquet
files no longer referenced by the Delta log. Without it, storage
grows unbounded. At 43GB/day ingest, unvacuumed storage would
cost ~$800/month vs ~$200/month vacuumed.

**Confluent CKU right-sizing.** 2 CKUs supports up to 2,000
msg/sec. At 1,000 txn/sec we're running at 50% utilisation.
Monitor and downsize to 1 CKU during off-peak hours using
Confluent's auto-scaling — saves ~$300/month.

**dbt Cloud vs self-hosted.** dbt Cloud Team plan costs
$100/month per seat. For a 3-person data team, self-hosting
dbt Core on AKS ($300/month) breaks even at 3 seats and
saves money at 4+ seats.

---

## Cost vs Build-from-Scratch Comparison

A fully custom streaming fraud detection system built without
this stack would require:

- Custom message queue infrastructure: ~$2,000/month
- Custom feature store: ~$1,500/month
- Custom model serving: ~$1,000/month
- Engineering time to build and maintain: 2 FTE = ~$30,000/month

**Total custom build: ~$34,500/month vs $5,500/month with
this stack.** The open-source lakehouse approach delivers
an 84% cost reduction versus a fully custom implementation.

---

## Scaling Characteristics

| Metric | Current (local) | Production | 10× Scale |
|--------|----------------|------------|-----------|
| Throughput | 10 txn/sec | 1,000 txn/sec | 10,000 txn/sec |
| Kafka brokers | 1 | 3 | 9 |
| Spark cores | 2 | 16 | 128 |
| Monthly cost | $0 | $5,500 | ~$28,000 |

At 10× scale (10,000 txn/sec), the dominant cost shifts to
Databricks compute and Confluent throughput. The architecture
scales horizontally without redesign — add brokers, add Spark
executors, increase Delta Lake partitioning.