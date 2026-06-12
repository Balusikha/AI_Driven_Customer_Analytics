# ==============================
# 03_train_prediction_model.py
# Purchase Prediction Model
# ==============================

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# Load Dataset
# ==============================
DATA_PATH = "data/processed_data.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)

# ==============================
# Target Column Creation
# ==============================
# If "purchase" column doesn't exist, create a synthetic one
if "purchase" not in df.columns:
    np.random.seed(42)
    df["purchase"] = (df["purchase_amount"] > df["purchase_amount"].median()).astype(int)

# ==============================
# Feature Selection
# ==============================
features = df.select_dtypes(include=[np.number])

# Remove unwanted columns
drop_cols = ["purchase", "segment", "churn"]
for col in drop_cols:
    if col in features.columns:
        features = features.drop(columns=[col])

X = features.fillna(features.mean())
y = df["purchase"]

# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    max_depth=10
)

model.fit(X_train_scaled, y_train)

# ==============================
# Predictions
# ==============================
y_pred = model.predict(X_test_scaled)

# ==============================
# Evaluation
# ==============================
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# Save Model + Scaler
# ==============================
os.makedirs("models", exist_ok=True)

with open("models/purchase_prediction.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/purchase_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Purchase prediction model saved successfully!")

# ==============================
# Feature Importance (for dashboard)
# ==============================
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

importance.to_csv("data/purchase_feature_importance.csv", index=False)

print("Feature importance saved.")