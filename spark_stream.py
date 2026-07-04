from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, when, lit, log1p, udf
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType, LongType, StructField
import json
import xgboost as xgb
import pandas as pd
import numpy as np
# from pyspark.sql.streaming.state import GroupStateTimeout
import sys
sys.path.insert(
    0,
    "/app/python_libs"
)
import redis
import time
import uuid
import psycopg2

spark = SparkSession.builder \
        .appName("FraudDetectionStreaming") \
        .config(
            "spark.executorEnv.PYTHONPATH",
            "/app/python_libs"
        ) \
        .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

redis_client = redis.Redis(
    host="redis",
    port=6379,
    decode_responses=True
)

print(
    "Redis connection established"
)

postgres_conn = psycopg2.connect(
    host="postgres",
    port=5432,
    database="frauddb",
    user="frauduser",
    password="fraudpass"
)

postgres_conn.autocommit = True
postgres_cursor = postgres_conn.cursor()
print(
    "PostgreSQL connection established"
)

# redis_client.set(
#     "test_key",
#     "hello"
# )
# print(
#     redis_client.get("test_key")
# )

model = xgb.Booster()
model.load_model("/app/models/fraud_model.json")
print("Fraud model loaded successfully")

with open(
    "/app/models/model_metadata.json",
    "r"
) as f:
    metadata = json.load(f)
print("Metadata loaded successfully")
# print(
#     "Threshold:",
#     metadata["threshold"]
# )
# print(
#     "Features:",
#     metadata["features"]
# )
MODEL_FEATURES = metadata["features"]

STEP_STATS = pd.read_csv(
    "/app/models/step_stats.csv"
)

STEP_LOOKUP = {
    int(row.step): {
        "system_volume_this_step": float(row.system_volume_this_step),
        "step_avg_amt": float(row.step_avg_amt)
    }
    for _, row in STEP_STATS.iterrows()
}
print("Step lookup loaded successfully")

def get_sender_features(sender):
    current_count = redis_client.get(
        f"sender:{sender}"
    )
    if current_count is None:
        sender_txn_count = 0
        is_first_sender_txn = 1
        redis_client.set(
            f"sender:{sender}",
            1
        )
    else:
        sender_txn_count = int(
            current_count
        )
        is_first_sender_txn = 0
        redis_client.incr(
            f"sender:{sender}"
        )
    return (
        sender_txn_count,
        is_first_sender_txn
    )

def get_receiver_features(receiver):
    current_count = redis_client.get(
        f"receiver:{receiver}"
    )
    if current_count is None:
        dest_txn_count = 0
        is_first_dest_txn = 1
        redis_client.set(
            f"receiver:{receiver}",
            1
        )
    else:
        dest_txn_count = int(
            current_count
        )
        is_first_dest_txn = 0
        redis_client.incr(
            f"receiver:{receiver}"
        )
    return (
        dest_txn_count,
        is_first_dest_txn
    )

def process_batch(batch_df, batch_id):

    rows = (
        batch_df
        .orderBy("row_id")
        .collect()
    )
    print(
        "\nRows in batch:", len(rows))

    print(f"\nProcessing Batch {batch_id}")

    for row in rows:
        try:
            if row.type not in [
                "TRANSFER",
                "CASH_OUT"
            ]:
                # prediction = "NORMAL"
                # fraud_probability = 0.0

                result = {
                    "step": row.step,
                    "type": row.type,
                    "amount": row.amount,
                    "origin": row.origin,
                    "destination": row.destination,
                    "actual_fraud": row.isFraud,
                    "fraud_probability": 0.0,
                    "prediction": "NORMAL"
                }

                print(result)

                postgres_cursor.execute(
                    """
                    INSERT INTO fraud_predictions (
                        step,
                        transaction_type,
                        amount,
                        origin,
                        destination,
                        actual_fraud,
                        fraud_probability,
                        prediction
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        result["step"],
                        result["type"],
                        result["amount"],
                        result["origin"],
                        result["destination"],
                        result["actual_fraud"],
                        result["fraud_probability"],
                        result["prediction"]
                    )
                )

                postgres_conn.commit()
                continue

            sender_txn_count, is_first_sender_txn = (
                get_sender_features(
                    row.origin
                )
            )

            dest_txn_count, is_first_dest_txn = (
                get_receiver_features(
                    row.destination
                )
            )

            step_info = STEP_LOOKUP.get(
                int(row.step)
            )

            if step_info:
                system_volume_this_step = (
                    step_info[
                        "system_volume_this_step"
                    ]
                )

                step_avg_amt = (
                    step_info[
                        "step_avg_amt"
                    ]
                )

            else:
                system_volume_this_step = 0.0
                step_avg_amt = row.amount

            amt_vs_step_avg = (
                row.amount/(step_avg_amt + 1)
            )

            print(
                "FEATURE_VECTOR:",
                [
                    row.amount,
                    row.log_amount,
                    row.is_transfer,
                    sender_txn_count,
                    is_first_sender_txn,
                    dest_txn_count,
                    is_first_dest_txn,
                    system_volume_this_step,
                    amt_vs_step_avg
                ]
            )

            print(
                f"ROW={row.row_id} "
                f"SENDER_COUNT={sender_txn_count} "
                f"DEST_COUNT={dest_txn_count}"
            )

            fraud_probability = predict_probability(
                row.amount,
                row.log_amount,
                row.is_transfer,
                sender_txn_count,
                is_first_sender_txn,
                dest_txn_count,
                is_first_dest_txn,
                system_volume_this_step,
                amt_vs_step_avg
            )

            prediction = (
                "FRAUD"
                if fraud_probability >= metadata["threshold"]
                else "NORMAL"
            )

            result = {
                "step": row.step,
                "type": row.type,
                "amount": row.amount,
                "origin": row.origin,
                "destination": row.destination,
                "actual_fraud": row.isFraud,
                "fraud_probability": float(
                    fraud_probability
                ),
                "prediction": prediction
            }

            print(result)

            postgres_cursor.execute(
                """
                INSERT INTO fraud_predictions (
                    step,
                    transaction_type,
                    amount,
                    origin,
                    destination,
                    actual_fraud,
                    fraud_probability,
                    prediction
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    result["step"],
                    result["type"],
                    result["amount"],
                    result["origin"],
                    result["destination"],
                    result["actual_fraud"],
                    result["fraud_probability"],
                    result["prediction"]
                )
            )
            print("INSERTED TO POSTGRES")

        except Exception as e:
            print("ERROR PROCESSING ROW:")
            print(e)

# sender_state_schema = StructType([
#     StructField(
#         "txn_count",
#         LongType(),
#         True
#     )
# ])

# sender_output_schema = StructType([
#     StructField("origin", StringType(), True),
#     StructField("destination", StringType(), True),
#     StructField("step", IntegerType(), True),
#     StructField("type", StringType(), True),
#     StructField("amount", DoubleType(), True),
#     StructField("isFraud", IntegerType(), True),

#     StructField("sender_txn_count", LongType(), True),
#     StructField("is_first_sender_txn", IntegerType(), True)
# ])

# def update_sender_state(
#     key,
#     pdf_iter,
#     state
# ):

#     if state.exists:
#         current_count = state.get()[0]
#     else:
#         current_count = 0

#     for pdf in pdf_iter:

#         sender_counts = []
#         first_flags = []

#         for _ in range(len(pdf)):

#             sender_counts.append(
#                 current_count
#             )

#             first_flags.append(
#                 1 if current_count == 0 else 0
#             )

#             current_count += 1

#         pdf["sender_txn_count"] = sender_counts

#         pdf["is_first_sender_txn"] = first_flags

#         yield pdf[
#             [
#                 "origin",
#                 "destination",
#                 "step",
#                 "type",
#                 "amount",
#                 "isFraud",
#                 "sender_txn_count",
#                 "is_first_sender_txn"
#             ]
#         ]

#     state.update((current_count,))

def predict_probability(
    amount,
    log_amount,
    is_transfer,
    sender_txn_count,
    is_first_sender_txn,
    dest_txn_count,
    is_first_dest_txn,
    system_volume_this_step,
    amt_vs_step_avg
):
        data = pd.DataFrame(
        [[
            amount,
            log_amount,
            is_transfer,
            sender_txn_count,
            is_first_sender_txn,
            dest_txn_count,
            is_first_dest_txn,
            system_volume_this_step,
            amt_vs_step_avg
        ]],
        columns=MODEL_FEATURES)
        
        dmatrix = xgb.DMatrix(data)
        probability = model.predict(dmatrix)[0]

        return float(probability)

transaction_schema = StructType() \
                    .add("row_id", LongType()) \
                    .add("step", IntegerType()) \
                    .add("type", StringType()) \
                    .add("amount", DoubleType()) \
                    .add("origin", StringType()) \
                    .add("destination", StringType()) \
                    .add("isFraud", IntegerType())

raw_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "transactions") \
    .option("startingOffsets", "latest") \
    .load()

json_df = raw_df.selectExpr(
    "CAST(value AS STRING) as json_data"
)

structured_df = json_df.select(
    from_json(
        col("json_data"),
        transaction_schema
    ).alias("data")
)

final_df = structured_df.select("data.*")

# final_df = final_df.filter(
#     col("type").isin(
#         "TRANSFER",
#         "CASH_OUT"
#     )
# )

final_df = final_df.withColumn(
    "log_amount",
    log1p(col("amount"))
)

final_df = final_df.withColumn(
    "is_transfer",
    when(col("type") == "TRANSFER",1).otherwise(0)
)

# sender_features_df = (
#     final_df
#     .groupBy("origin")
#     .applyInPandasWithState(
#         func=update_sender_state,
#         outputStructType=sender_output_schema,
#         stateStructType=sender_state_schema,
#         outputMode="append",
#         timeoutConf=GroupStateTimeout.NoTimeout
#     )
# )

# query = final_df.writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .option("truncate", "false") \
#     .start()

query = (
    final_df.writeStream
    .foreachBatch(
        process_batch
    )
    .start()
)

print("Streaming query started")

query.awaitTermination()
