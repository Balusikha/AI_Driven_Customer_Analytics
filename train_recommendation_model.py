# ==============================
# 04_train_recommendation_model.py
# Recommendation System (Customer Similarity Based)
# ==============================

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ==============================
# Load Dataset
# ==============================
DATA_PATH = "data/processed_data.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)

# ==============================
# Feature Engineering
# ==============================
# We build customer-level features for similarity

if "customer_id" not in df.columns:
    df["customer_id"] = np.arange(len(df))

# Aggregate numeric behavior per customer
numeric_cols = df.select_dtypes(include=[np.number]).columns

customer_features = df.groupby("customer_id")[numeric_cols].mean()

# Fill missing values
customer_features = customer_features.fillna(customer_features.mean())

# ==============================
# Standardization
# ==============================
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features)

# ==============================
# Similarity Matrix
# ==============================
similarity_matrix = cosine_similarity(scaled_features)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=customer_features.index,
    columns=customer_features.index
)

print("Similarity matrix created:", similarity_df.shape)

# ==============================
# Recommendation Function Logic
# ==============================
def get_recommendations(customer_id, top_n=5):
    if customer_id not in similarity_df.index:
        return []

    similar_customers = similarity_df[customer_id].sort_values(ascending=False)

    # Remove self similarity
    similar_customers = similar_customers.drop(customer_id)

    top_customers = similar_customers.head(top_n).index.tolist()

    # Generate dummy product recommendations (can be replaced later with real product mapping)
    recommendations = [
        f"Product_{i}" for i in range(1, top_n + 1)
    ]

    return {
        "similar_customers": top_customers,
        "recommendations": recommendations
    }

# ==============================
# Save Model Artifacts
# ==============================
os.makedirs("models", exist_ok=True)

with open("models/recommendation_model.pkl", "wb") as f:
    pickle.dump(similarity_df, f)

with open("models/recommendation_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Recommendation model saved successfully!")

# ==============================
# Save customer feature table
# ==============================
customer_features.to_csv("data/customer_features.csv")

print("Customer feature matrix saved.")