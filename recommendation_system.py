# ==============================
# 08_recommendation_system.py
# Hybrid Recommendation Engine
# ==============================

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ==============================
# Load Data
# ==============================
DATA_PATH = "data/processed_data.csv"

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

df = load_data()

# ==============================
# Load Precomputed Similarity Model (optional)
# ==============================
MODEL_PATH = "models/recommendation_model.pkl"

def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


similarity_matrix = load_pickle(MODEL_PATH)

# ==============================
# Feature Engineering
# ==============================
def build_customer_matrix(df):
    if "customer_id" not in df.columns:
        df["customer_id"] = np.arange(len(df))

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    customer_matrix = df.groupby("customer_id")[numeric_cols].mean()
    customer_matrix = customer_matrix.fillna(customer_matrix.mean())

    return customer_matrix

# ==============================
# Similarity Computation
# ==============================
def compute_similarity_matrix(customer_matrix):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(customer_matrix)

    similarity = cosine_similarity(scaled)

    return pd.DataFrame(
        similarity,
        index=customer_matrix.index,
        columns=customer_matrix.index
    )

# ==============================
# Recommendation Engine (Core)
# ==============================
def recommend_products(customer_id, top_n=5, df_input=None):
    """
    Hybrid recommendation:
    - similarity-based customers
    - fallback popularity
    """

    df_input = df_input if df_input is not None else df

    if df_input.empty:
        return {"error": "No data available"}

    customer_matrix = build_customer_matrix(df_input)

    # Use precomputed similarity OR compute fresh
    if similarity_matrix is not None:
        sim_df = similarity_matrix
    else:
        sim_df = compute_similarity_matrix(customer_matrix)

    if customer_id not in sim_df.index:
        return {"error": "Customer not found"}

    # Similar customers
    similar_customers = sim_df[customer_id].sort_values(ascending=False)
    similar_customers = similar_customers.drop(customer_id, errors="ignore")

    top_similar = similar_customers.head(top_n).index.tolist()

    # ==============================
    # Product Recommendation Logic
    # ==============================
    if "product_id" in df_input.columns:
        product_map = df_input[df_input["customer_id"].isin(top_similar)]

        top_products = (
            product_map["product_id"]
            .value_counts()
            .head(top_n)
            .index.tolist()
        )
    else:
        # fallback synthetic products
        top_products = [f"Product_{i}" for i in range(1, top_n + 1)]

    return {
        "customer_id": customer_id,
        "similar_customers": top_similar,
        "recommended_products": top_products
    }

# ==============================
# Popularity-Based Fallback
# ==============================
def get_popular_products(df_input, top_n=5):
    if "product_id" not in df_input.columns:
        return [f"Product_{i}" for i in range(1, top_n + 1)]

    return (
        df_input["product_id"]
        .value_counts()
        .head(top_n)
        .index.tolist()
    )

# ==============================
# Batch Recommendations
# ==============================
def batch_recommend(customer_ids, top_n=5):
    results = []

    for cid in customer_ids:
        results.append(recommend_products(cid, top_n))

    return results

# ==============================
# Insight Generator
# ==============================
def recommendation_insights(df_input):
    popular = get_popular_products(df_input)

    return {
        "total_customers": df_input["customer_id"].nunique() if "customer_id" in df_input.columns else len(df_input),
        "top_products": popular
    }

# ==============================
# Test Run
# ==============================
if __name__ == "__main__":
    if not df.empty:
        print(recommend_products(1))
        print(recommendation_insights(df))