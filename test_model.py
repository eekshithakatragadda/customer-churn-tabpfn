import json
import functools
import joblib
import pandas as pd
import torch

from pathlib import Path

base = Path(__file__).parent

# Load model with CPU mapping
original_load = torch.load
try:
    torch.load = functools.partial(
        original_load,
        map_location="cpu",
        weights_only=False
    )
    model = joblib.load(
        base / "artifacts" / "tabpfn_churn_model.pkl"
    )
finally:
    torch.load = original_load

model.device = "cpu"
model.devices_ = (torch.device("cpu"),)
model.keep_cache_on_device = False

# Prepare one real customer
df = pd.read_csv(base / "customer_churn.csv")
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
).fillna(0)

with open(base / "artifacts" / "feature_columns.json") as f:
    columns = json.load(f)

X = df.drop(columns=["customerID", "Churn"], errors="ignore")
X = X[columns]

customer = X.iloc[[0]].copy()

print("Testing one customer...")
print("Model device:", model.device)
print("Input shape:", customer.shape)

print("Prediction:", model.predict(customer))
print("Churn probabilities:", model.predict_proba(customer))