# ==============================
# 09_churn_analysis.py
# Churn Analytics & Insights Engine
# ==============================

import pandas as pd
import numpy as np
import pickle
import os

# ==============================
# Load Dataset
# ==============================
DATA_PATH = "data/churn_scored_data.csv"

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

df = load_data()

# ==============================
# Load Model (optional use)
# ==============================
MODEL_PATH = "models/churn_model.pkl"

def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


model = load_pickle(MODEL_PATH)

# ==============================
# Risk Level Classification
# ==============================
def assign_risk_level(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

# ==============================
# Enrich Dataset with Risk Labels
# ==============================
def enrich_churn_data(df_input):
    df_input = df_input.copy()

    if "churn_probability" not in df_input.columns:
        return df_input

    df_input["risk_level"] = df_input["churn_probability"].apply(assign_risk_level)

    return df_input

# ==============================
# Churn Summary KPIs
# ==============================
def churn_kpis(df_input):
    if df_input.empty:
        return {
            "total_customers": 0,
            "churn_rate": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0
        }

    df_input = enrich_churn_data(df_input)

    total = len(df_input)

    churn_rate = df_input["churn_probability"].mean() if "churn_probability" in df_input.columns else 0

    risk_counts = df_input["risk_level"].value_counts().to_dict()

    return {
        "total_customers": int(total),
        "churn_rate": float(churn_rate),
        "high_risk": int(risk_counts.get("High Risk", 0)),
        "medium_risk": int(risk_counts.get("Medium Risk", 0)),
        "low_risk": int(risk_counts.get("Low Risk", 0))
    }

# ==============================
# High Risk Customers Extractor
# ==============================
def get_high_risk_customers(df_input, top_n=20):
    df_input = enrich_churn_data(df_input)

    if "churn_probability" not in df_input.columns:
        return pd.DataFrame()

    high_risk = df_input.sort_values(by="churn_probability", ascending=False)

    return high_risk.head(top_n)[
        ["customer_id", "churn_probability", "risk_level"]
        if "customer_id" in df_input.columns else ["churn_probability", "risk_level"]
    ]

# ==============================
# Segment-wise Churn Analysis
# ==============================
def segment_churn_analysis(df_input):
    df_input = enrich_churn_data(df_input)

    if "segment" not in df_input.columns:
        df_input["segment"] = 0

    analysis = df_input.groupby("segment").agg({
        "churn_probability": "mean"
    }).rename(columns={"churn_probability": "avg_churn_risk"})

    return analysis.reset_index()

# ==============================
# Retention Insights Generator
# ==============================
def retention_insights(df_input):
    df_input = enrich_churn_data(df_input)

    insights = {}

    insights["total_customers"] = len(df_input)
    insights["avg_churn_risk"] = float(df_input["churn_probability"].mean()) if "churn_probability" in df_input.columns else 0

    # Identify most risky segment
    if "segment" in df_input.columns:
        risky_segment = df_input.groupby("segment")["churn_probability"].mean().idxmax()
        insights["most_risky_segment"] = int(risky_segment)

    # Top churn drivers (approx using numeric correlation)
    numeric_df = df_input.select_dtypes(include=[np.number])
    if "churn_probability" in numeric_df.columns:
        corr = numeric_df.corr()["churn_probability"].sort_values(ascending=False)
        insights["top_risk_factors"] = corr.head(5).index.tolist()

    return insights

# ==============================
# Export Dashboard Dataset
# ==============================
def export_dashboard_data(df_input):
    df_input = enrich_churn_data(df_input)

    os.makedirs("data", exist_ok=True)

    df_input.to_csv("data/churn_dashboard_data.csv", index=False)

    return "data/churn_dashboard_data.csv"

# ==============================
# TEST RUN
# ==============================
if __name__ == "__main__":
    if not df.empty:
        print("Churn KPIs:")
        print(churn_kpis(df))

        print("\nHigh Risk Customers:")
        print(get_high_risk_customers(df).head())

        print("\nRetention Insights:")
        print(retention_insights(df))

        export_dashboard_data(df)