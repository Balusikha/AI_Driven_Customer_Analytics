# ==============================
# 02_train_segmentation_model.py
# Customer Segmentation using KMeans + PCA
# ==============================

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================
# Load Dataset
# ==============================
DATA_PATH = "data/processed_data.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)

# ==============================
# Feature Selection
# ==============================
# Keep only numeric columns for clustering
features = df.select_dtypes(include=[np.number])

# Remove target-like columns if exist
drop_cols = ["churn", "segment"]
for col in drop_cols:
    if col in features.columns:
        features = features.drop(columns=[col])

# Fill missing values
features = features.fillna(features.mean())

# ==============================
# Standardization
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ==============================
# KMeans Clustering
# ==============================
k = 4  # you can tune this later

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_scaled)

print("Clustering completed with", k, "segments")

# ==============================
# PCA for Visualization
# ==============================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df["pca_x"] = pca_result[:, 0]
df["pca_y"] = pca_result[:, 1]

# ==============================
# Save Processed Dataset
# ==============================
os.makedirs("data", exist_ok=True)
df.to_csv("data/processed_data.csv", index=False)

# ==============================
# Save Models
# ==============================
os.makedirs("models", exist_ok=True)

with open("models/customer_segmentation.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/pca.pkl", "wb") as f:
    pickle.dump(pca, f)

print("Models saved successfully!")

# ==============================
# Cluster Summary Report
# ==============================
cluster_summary = df.groupby("segment").mean(numeric_only=True)

print("\nCluster Summary:")
print(cluster_summary)

# Save summary for dashboard usage
cluster_summary.to_csv("data/cluster_summary.csv")

print("Segmentation training completed.")