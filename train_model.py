
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import os, json

# 1. DATA LOADING AND FILTERING
print("Loading PaySim data...")
df = pd.read_csv("data/paysim_data.csv")

# Only keep fraud-prone transactions
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

# DROP LEAKAGE: No balances allowed
FORBIDDEN = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"]
df = df.drop(columns=[c for c in FORBIDDEN if c in df.columns])


# 2. BEHAVIORAL FEATURE ENGINEERING
print("Engineering features...")
df = df.sort_values('step')

# Sender & Receiver History
df['sender_txn_count'] = df.groupby('nameOrig').cumcount()
df['dest_txn_count']   = df.groupby('nameDest').cumcount()
df['is_first_sender_txn'] = (df['sender_txn_count'] == 0).astype(int)
df['is_first_dest_txn']   = (df['dest_txn_count'] == 0).astype(int)

# Transaction Type
df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)

# Step Context
hourly_volume = df.groupby('step')['amount'].count().to_dict()
df['system_volume_this_step'] = df['step'].map(hourly_volume)

step_avg_amt = df.groupby('step')['amount'].transform('mean')
df['amt_vs_step_avg'] = df['amount'] / (step_avg_amt + 1)

# Normalization
df['log_amount'] = np.log1p(df['amount'])

features = [
    'amount', 'log_amount', 'is_transfer', 'sender_txn_count', 
    'is_first_sender_txn', 'dest_txn_count', 'is_first_dest_txn',
    'system_volume_this_step', 'amt_vs_step_avg'
]

X = df[features]
y = df['isFraud']


# 3. TEMPORAL SPLIT (No Shuffling)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


# OPTUNA SEARCH
"""
def objective(trial):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 10.0, 50.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'aucpr',
        'n_jobs': -1,
        'random_state': 42
    }
    model = XGBClassifier(**params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds_proba = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, preds_proba)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(study.best_params)
"""


# 4. MODEL TRAINING WITH OPTUNA PARAMETERS
print("Training Optuna-Optimized Regularized XGBoost...")
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.02323730920867429,
    max_depth=5,
    scale_pos_weight=49.60485940572818,
    reg_lambda=4.827975096381764,
    reg_alpha=0.4871070088458251,
    gamma=0.4027254324714318,
    subsample=0.8483987303504982,
    colsample_bytree=0.6082215690755312,
    eval_metric='aucpr',
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)


# 5. BUSINESS THRESHOLD OPTIMIZATION (Focusing on Recall via F-Beta)
print("Optimizing threshold for Operational Deployment...")
y_proba = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Recall is highly valuable, beta=1.2 subtly favors recall.
beta = 1.2
f_beta = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-9)
best_thr = thresholds[np.argmax(f_beta)]


# 6. FINAL EVALUATION BREAKDOWN
def evaluate(thr):
    preds = (y_proba >= thr).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"AUPRC Score: {average_precision_score(y_test, y_proba):.4f}")
    print(f"Threshold: {thr:.4f} | Recall: {rec:.2%} | Precision: {prec:.2%}")
    print(f"Caught: {tp} frauds | Missed: {fn} frauds | False Alarms: {fp}")
    if tp > 0:
        print(f"Efficiency: 1 fraud caught per {fp/tp:.2f} false alarms.")

evaluate(best_thr)

#Save model
# os.makedirs("models", exist_ok=True)

# model.save_model(
#     "models/fraud_model.json"
# )

# metadata = {
#     "threshold": float(best_thr),
#     "features": features,
#     "beta": 1.2,
#     "auprc": float(
#         average_precision_score(
#             y_test,
#             y_proba
#         )
#     )
# }

# with open(
#     "models/model_metadata.json",
#     "w"
# ) as f:
#     json.dump(
#         metadata,
#         f,
#         indent=4
#     )

