from kafka import KafkaProducer
import json
import time
import pandas as pd
import numpy as np

producer = KafkaProducer(
    bootstrap_servers = 'localhost:29092',
    value_serializer = lambda v: json.dumps(v).encode('utf-8')
)


df = pd.read_csv(
    'data/paysim_data.csv'
)

df["row_id"] = np.arange(len(df))

# df = df.head(100)

for _, row in df.iterrows():
    transaction = {
    "row_id": int(row["row_id"]),
    "step": int(row["step"]),
    "type": row["type"],
    "amount": float(row["amount"]),
    "origin": row["nameOrig"],
    "destination": row["nameDest"],
    "isFraud": int(row["isFraud"])
}


    future = producer.send(
        'transactions',
        value=transaction
    )

    record_metadata = future.get(timeout=10)

    print(
        f"Sent transaction to partition "
        f"{record_metadata.partition}, "
        f"offset {record_metadata.offset}"
    )
producer.flush()