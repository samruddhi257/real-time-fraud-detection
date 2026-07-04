import json
import xgboost as xgb
import pandas as pd

#Load trained model

# Create empty XGBoost booster object
model = xgb.Booster()
model.load_model("models/fraud_model.json")

print("Model loaded successfully")

# Metadata contains:
# threshold
# feature order 
# training metrics

with open(
    "models/model_metadata.json",
    "r"
) as f:
    metadata = json.load(f)

sample_transaction = {
    "amount": 500000,
    "log_amount": 13.122365,
    "is_transfer": 1,
    "sender_txn_count": 10,
    "is_first_sender_txn": 0,
    "dest_txn_count": 5,
    "is_first_dest_txn": 0,
    "system_volume_this_step": 3000,
    "amt_vs_step_avg": 4.2
}

#Prediction func
def predict_transaction(features_dict):
    # Convert dictionary to dataframe
    sample_df = pd.DataFrame(
        [features_dict]
    )

    # Enforce exact training feature order
    sample_df = sample_df[metadata["features"]]

    # Convert dataframe to XGBoost format
    dtest = xgb.DMatrix(sample_df)

    # Generate fraud probability
    fraud_probability = model.predict(dtest)[0]

    # Apply optimized business threshold
    decision = (
        "FRAUD" if fraud_probability>=metadata["threshold"]
        else "NORMAL"
    )

    return {
        "fraud_probability":float(fraud_probability), 
        "decision": decision
        }

# Dummy transaction for testing inference
sample_transaction = {
    "amount": 500000,
    "log_amount": 13.122365,
    "is_transfer": 1,
    "sender_txn_count": 10,
    "is_first_sender_txn": 0,
    "dest_txn_count": 5,
    "is_first_dest_txn": 0,
    "system_volume_this_step": 3000,
    "amt_vs_step_avg": 4.2
}

#Run Prediction
result = predict_transaction(
    sample_transaction
)

print("\nPrediction Result")
print(result)

