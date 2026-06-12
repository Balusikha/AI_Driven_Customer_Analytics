# ==============================
# 10_utils/data_preprocessing.py
# Unified Data Preprocessing Pipeline
# ==============================

import pandas as pd
import numpy as np
import os

# ==============================
# Load Dataset
# ==============================
def load_data(path="data/processed_data.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# ==============================
# Missing Value Handling
# ==============================
def handle_missing_values(df):
    df = df.copy()

    # Numeric columns → mean fill
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # Categorical columns → mode fill
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    return df

# ==============================
# Feature Engineering
# ==============================
def feature_engineering(df):
    df = df.copy()

    # ------------------------------
    # Purchase behavior features
    # ------------------------------
    if "purchase_amount" in df.columns:
        df["log_purchase"] = np.log1p(df["purchase_amount"])

        df["high_value_customer"] = (df["purchase_amount"] > df["purchase_amount"].median()).astype(int)

    # ------------------------------
    # Engagement score (synthetic feature)
    # ------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df["engagement_score"] = df[numeric_cols].mean(axis=1)

    # ------------------------------
    # Recency-like feature (if available)
    # ------------------------------
    if "last_purchase_days" in df.columns:
        df["recency_score"] = 1 / (1 + df["last_purchase_days"])

    return df

# ==============================
# Outlier Handling
# ==============================
def remove_outliers(df, threshold=3):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()

        df = df[(df[col] >= mean - threshold * std) &
                (df[col] <= mean + threshold * std)]

    return df

# ==============================
# Full Preprocessing Pipeline
# ==============================
def preprocess_data(df, remove_outlier_flag=False):
    """
    Main pipeline used across ALL models
    """

    df = handle_missing_values(df)
    df = feature_engineering(df)

    if remove_outlier_flag:
        df = remove_outliers(df)

    df = df.reset_index(drop=True)

    return df

# ==============================
# Train-Test Feature Split Helper
# ==============================
def get_features_target(df, target_col):
    """
    Splits dataset into X, y safely
    """

    df = df.copy()

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # keep only numeric for ML safety
    X = X.select_dtypes(include=[np.number])

    return X, y

# ==============================
# Save Processed Data
# ==============================
def save_processed_data(df, path="data/processed_data.csv"):
    os.makedirs("data", exist_ok=True)
    df.to_csv(path, index=False)
    return path

# ==============================
# TEST RUN
# ==============================
if __name__ == "__main__":
    df = load_data()

    if not df.empty:
        cleaned = preprocess_data(df)
        print("Original shape:", df.shape)
        print("Cleaned shape:", cleaned.shape)

        save_processed_data(cleaned)
        print("Data preprocessing completed.")