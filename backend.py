# =============================================================
# CREDIT CARD FRAUD DETECTION - FASTAPI BACKEND
# =============================================================

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier

app = FastAPI(title="Credit Card Fraud Detection API")

# -------------------------
# LOAD MODELS
# -------------------------
BASE_PATH = r'D:\nss kdd'

autoencoder = tf.keras.models.load_model(BASE_PATH + r'\autoencoder_fraud.keras')

with open(BASE_PATH + r'\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(BASE_PATH + r'\target_encoders.pkl', 'rb') as f:
    target_encoders = pickle.load(f)

with open(BASE_PATH + r'\best_threshold.pkl', 'rb') as f:
    best_threshold = pickle.load(f)

xgb_model = XGBClassifier()
xgb_model.load_model(BASE_PATH + r'\xgb_model.json')

print("All models loaded!")

# -------------------------
# FEATURE COLUMNS ORDER
# -------------------------
FEATURE_COLS = ['merchant', 'category', 'amt', 'gender', 'state', 'city_pop',
                'job', 'hour', 'day_of_week', 'age', 'distance',
                'tx_velocity', 'amt_deviation', 'tx_per_day']

# -------------------------
# REQUEST SCHEMA
# -------------------------
class Transaction(BaseModel):
    amt: float
    category: str
    merchant: str
    gender: str
    state: str
    city_pop: int
    job: str
    hour: int
    day_of_week: int
    age: int
    distance: float
    tx_velocity: int
    amt_deviation: float
    tx_per_day: int

# -------------------------
# ENDPOINTS
# -------------------------
@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is running!"}


@app.get("/categories")
def get_categories():
    return {
        "categories": list(target_encoders['category'].index.tolist()),
        "states": list(target_encoders['state'].index.tolist()),
        "jobs": list(target_encoders['job'].index.tolist()),
        "merchants": list(target_encoders['merchant'].index.tolist())
    }


@app.post("/predict")
def predict(transaction: Transaction):
    # Build raw dataframe
    data = {
        'merchant': transaction.merchant,
        'category': transaction.category,
        'amt': transaction.amt,
        'gender': 1 if transaction.gender == 'F' else 0,
        'state': transaction.state,
        'city_pop': transaction.city_pop,
        'job': transaction.job,
        'hour': transaction.hour,
        'day_of_week': transaction.day_of_week,
        'age': transaction.age,
        'distance': transaction.distance,
        'tx_velocity': transaction.tx_velocity,
        'amt_deviation': transaction.amt_deviation,
        'tx_per_day': transaction.tx_per_day
    }

    df = pd.DataFrame([data])

    # Target encode categorical columns
    cat_cols = ['job', 'category', 'merchant', 'state']
    for col in cat_cols:
        df[col] = df[col].map(target_encoders[col])
        df[col] = df[col].fillna(target_encoders[col].mean())

    # Reorder columns
    df = df[FEATURE_COLS]

    # Scale
    df_scaled = pd.DataFrame(scaler.transform(df), columns=FEATURE_COLS)

    # Get reconstruction error
    pred_ae = autoencoder.predict(df_scaled, verbose=0)
    recon_error = float(np.mean(np.power(df_scaled.values - pred_ae, 2)))

    # Add recon error to features
    df_scaled['recon_error'] = recon_error

    # XGBoost prediction
    fraud_prob = float(xgb_model.predict_proba(df_scaled)[:, 1][0])
    is_fraud = int(fraud_prob > best_threshold)

    return {
        "prediction": "Fraud" if is_fraud else "Legitimate",
        "fraud_probability": round(fraud_prob * 100, 2),
        "reconstruction_error": round(recon_error, 4),
        "threshold_used": round(float(best_threshold), 4)
    }
