# ==============================
# 05_train_churn_model.py
# Customer Churn Prediction Model
# ==============================

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# ==============================
# Load Dataset
# ==============================
DATA_PATH = "data/processed_data.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)

# ==============================
# Create Churn Column (if not exists)
# ==============================
# If real churn column is missing, create a realistic proxy
if "churn" not in df.columns:
    np.random.seed(42)

    # churn logic based on low purchase + low activity
    threshold = df["purchase_amount"].median() if "purchase_amount" in df.columns else 0

    df["churn"] = (
        (df["purchase_amount"] < threshold) |
        (df.select_dtypes(include=[np.number]).mean(axis=1) < 0.3)
    ).astype(int)

# ==============================
# Feature Selection
# ==============================
features = df.select_dtypes(include=[np.number])

# Remove target column
if "churn" in features.columns:
    features = features.drop(columns=["churn"])

X = features.fillna(features.mean())
y = df["churn"]

# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# Feature Scaling
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Model Training
# ==============================
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

# ==============================
# Predictions
# ==============================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ==============================
# Evaluation
# ==============================
accuracy = accuracy_score(y_test, y_pred)

print("\nChurn Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# Feature Importance (Coefficients)
# ==============================
importance = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
}).sort_values(by="coefficient", ascending=False)

# ==============================
# Save Model
# ==============================
os.makedirs("models", exist_ok=True)

with open("models/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/churn_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Churn model saved successfully!")

# ==============================
# Save Analytics Outputs
# ==============================
df["churn_probability"] = model.predict_proba(X)[:, 1]

df.to_csv("data/churn_scored_data.csv", index=False)

importance.to_csv("data/churn_feature_importance.csv", index=False)

print("Churn analytics files saved successfully!")