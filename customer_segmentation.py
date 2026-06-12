# ==============================
# 06_customer_segmentation.py
# Customer Segmentation Inference Module
# ==============================

import pandas as pd
import numpy as np
import pickle
import os

# ==============================
# Load Models
# ==============================
MODEL_PATH = "models/customer_segmentation.pkl"
SCALER_PATH = "models/scaler.pkl"
PCA_PATH = "models/pca.pkl"

def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


kmeans_model = load_pickle(MODEL_PATH)
scaler = load_pickle(SCALER_PATH)
pca = load_pickle(PCA_PATH)

# ==============================
# Load Data (optional fallback usage)
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
    numeric_df = input_df.select_dtypes(include=[np.number])

    # Remove unwanted columns if exist
    drop_cols = ["segment", "churn", "pca_x", "pca_y"]
    numeric_df = numeric_df.drop(columns=[col for col in drop_cols if col in numeric_df.columns])

    numeric_df = numeric_df.fillna(numeric_df.mean())

    return numeric_df

# ==============================
# Predict Segment for New Data
# ==============================
def predict_segment(input_data):
    """
    input_data: pandas DataFrame or dict
    """

    if kmeans_model is None:
        return {"error": "Model not loaded"}

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    features = prepare_features(input_data)

    if scaler:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features

    segments = kmeans_model.predict(features_scaled)

    return {
        "segments": segments.tolist()
    }

# ==============================
# Add Segment + PCA to Dataset
# ==============================
def enrich_dataset(df):
    """
    Adds:
    - segment labels
    - PCA coordinates (for visualization)
    """

    features = prepare_features(df)

    if scaler:
        scaled = scaler.transform(features)
    else:
        scaled = features

    df = df.copy()

    if kmeans_model:
        df["segment"] = kmeans_model.predict(scaled)

    # PCA transformation for visualization
    if pca:
        pca_result = pca.transform(scaled)
        df["pca_x"] = pca_result[:, 0]
        df["pca_y"] = pca_result[:, 1]

    return df

# ==============================
# Segment Summary Analytics
# ==============================
def get_segment_summary(df):
    if "segment" not in df.columns:
        df = enrich_dataset(df)

    summary = df.groupby("segment").mean(numeric_only=True)
    return summary

# ==============================
# Segment Insights Generator
# ==============================
def segment_insights(df):
    if "segment" not in df.columns:
        df = enrich_dataset(df)

    insights = {}

    for seg in df["segment"].unique():
        seg_data = df[df["segment"] == seg]

        insights[int(seg)] = {
            "count": len(seg_data),
            "avg_purchase": float(seg_data["purchase_amount"].mean()) if "purchase_amount" in df.columns else 0,
            "avg_churn": float(seg_data["churn"].mean()) if "churn" in df.columns else 0
        }

    return insights

# ==============================
# Test Run (optional)
# ==============================
if __name__ == "__main__":
    test_df = load_data()

    if not test_df.empty:
        enriched = enrich_dataset(test_df.head(10))
        print(enriched[["segment", "pca_x", "pca_y"]].head())

        print("\nSegment Insights:")
        print(segment_insights(test_df))