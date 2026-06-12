# ==============================
# 07_purchase_prediction.py
# Purchase Prediction Inference Engine
# ==============================

import pandas as pd
import numpy as np
import pickle
import os

# ==============================
# Load Model + Scaler
# ==============================
MODEL_PATH = "models/purchase_prediction.pkl"
SCALER_PATH = "models/purchase_scaler.pkl"

def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


model = load_pickle(MODEL_PATH)
scaler = load_pickle(SCALER_PATH)

# ==============================
# Load Dataset (optional)
# ==============================
DATA_PATH = "data/processed_data.csv"

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

df = load_data()

# ==============================
# Feature Preparation
# ==============================
def prepare_features(input_df):
    """
    Converts raw input into ML-ready numeric features
    """

    if isinstance(input_df, dict):
        input_df = pd.DataFrame([input_df])

    features = input_df.select_dtypes(include=[np.number])

    # Remove target/leakage columns if present
    drop_cols = ["purchase", "segment", "churn"]
    features = features.drop(columns=[col for col in drop_cols if col in features.columns])

    features = features.fillna(features.mean())

    return features

# ==============================
# Predict Purchase (Binary)
# ==============================
def predict_purchase(input_data):
    """
    Returns:
    - prediction (0/1)
    - probability score
    """

    if model is None:
        return {"error": "Model not loaded"}

    features = prepare_features(input_data)

    if scaler:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features

    prediction = model.predict(features_scaled)[0]

    # Probability (if supported)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features_scaled)[0][1]
    else:
        probability = None

    return {
        "prediction": int(prediction),
        "purchase_probability": float(probability) if probability is not None else None
    }

# ==============================
# Batch Prediction (Dashboard Use)
# ==============================
def batch_predict(df_input):
    """
    Predict multiple customers at once
    """

    features = prepare_features(df_input)

    if scaler:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features

    predictions = model.predict(features_scaled)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_scaled)[:, 1]

    result = df_input.copy()
    result["purchase_prediction"] = predictions

    if probabilities is not None:
        result["purchase_probability"] = probabilities

    return result

# ==============================
# Insight Generator
# ==============================
def prediction_insights(df_input):
    """
    Business insights for dashboard
    """

    results = batch_predict(df_input)

    total = len(results)
    buyers = results["purchase_prediction"].sum()

    avg_prob = results["purchase_probability"].mean() if "purchase_probability" in results.columns else 0

    return {
        "total_customers": int(total),
        "predicted_buyers": int(buyers),
        "conversion_rate": float(buyers / total) if total > 0 else 0,
        "avg_purchase_probability": float(avg_prob)
    }

# ==============================
# Test Run
# ==============================
if __name__ == "__main__":
    if not df.empty:
        sample = df.head(5)
        print("Single Prediction:")
        print(predict_purchase(sample.iloc[0]))

        print("\nBatch Insights:")
        print(prediction_insights(sample))