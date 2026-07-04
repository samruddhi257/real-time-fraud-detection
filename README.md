# Real-Time Fraud Detection System

## Overview

A real-time fraud detection pipeline built using Apache Kafka, Spark Structured Streaming, Redis, XGBoost, PostgreSQL, and Power BI. The system streams financial transactions from the PaySim dataset, performs real-time feature engineering, predicts fraudulent transactions using a trained XGBoost model, stores predictions in PostgreSQL, and visualizes them through an interactive Power BI dashboard.

---

## Features

- Streams financial transactions using Apache Kafka.
- Performs real-time feature engineering with Redis.
- Detects fraudulent transactions using an XGBoost model.
- Stores predictions in PostgreSQL.
- Visualizes fraud analytics through Power BI.
- Supports continuous processing of large transaction streams.


---

## Tech Stack

| Component | Technology |
|----------|------------|
| Language | Python |
| Streaming | Apache Kafka |
| Processing | Spark Structured Streaming |
| Machine Learning | XGBoost |
| State Store | Redis |
| Database | PostgreSQL |
| Dashboard | Power BI |
| Containerization | Docker |

---

## Project Structure

```
.
├── data/
├── models/
├── dashboard/
├── producer.py
├── spark_stream.py
├── train_model.py
├── docker-compose.yml
├── Dockerfile.spark
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/samruddhi257/real-time-fraud-detection.git
cd real-time-fraud-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Docker services

```bash
docker compose up -d
```

---

## Running the Project

### Train the model

```bash
python train_model.py
```

### Start Spark Streaming

```bash
docker exec -it spark bash

spark-submit spark_stream.py
```

### Start the Kafka Producer

```bash
python producer.py
```

The streaming application continuously reads transactions from Kafka, generates features, predicts fraud, stores predictions in PostgreSQL, and updates the Power BI dashboard.

---

## Model Performance

| Metric | Value |
|--------|------:|
| AUPRC | 0.8108 |
| Precision | 89.73% |
| Recall | 70.99% |

---

## Dashboard

The Power BI dashboard provides real-time monitoring of incoming transactions and fraud predictions.

It includes:

- Total Transactions
- Frauds Detected
- Fraud Rate
- Average Fraud Score
- Fraud vs Normal Distribution
- Fraud Count by Transaction Type
- Top Suspicious Accounts
- High Risk Transactions Table

---

## Future Improvements

- Integrate Apache Flink for event-by-event stream processing.
- Deploy the pipeline on Kubernetes.
- Add model monitoring and drift detection.
- Replace manual dashboard refresh with automatic live visualization.

---

## Dataset

The project uses the PaySim synthetic financial transaction dataset for fraud detection research. The dataset is available on Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1
