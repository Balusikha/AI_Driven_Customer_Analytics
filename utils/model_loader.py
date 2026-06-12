# ==============================
# 12_utils/model_loader.py
# Centralized Model Loader (Production Safe)
# ==============================

import os
import pickle

# ==============================
# Generic Model Loader
# ==============================
def load_model(model_path):
    """
    Safely loads a pickle model.
    Returns None if file not found or corrupted.
    """

    if not os.path.exists(model_path):
        print(f"[WARNING] Model not found: {model_path}")
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print(f"[INFO] Loaded model: {model_path}")
        return model

    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}")
        return None

# ==============================
# Model Registry (Central Access Point)
# ==============================
class ModelRegistry:
    """
    Centralized access for all ML models
    """

    def __init__(self):
        self.models = {}

    def register(self, name, path):
        model = load_model(path)
        self.models[name] = model

    def get(self, name):
        return self.models.get(name, None)

    def reload(self, name, path):
        self.models[name] = load_model(path)

    def list_models(self):
        return list(self.models.keys())

# ==============================
# Predefined Model Paths
# ==============================
MODEL_PATHS = {
    "segmentation": "models/customer_segmentation.pkl",
    "scaler": "models/scaler.pkl",
    "pca": "models/pca.pkl",
    "purchase_model": "models/purchase_prediction.pkl",
    "purchase_scaler": "models/purchase_scaler.pkl",
    "churn_model": "models/churn_model.pkl",
    "churn_scaler": "models/churn_scaler.pkl",
    "recommendation": "models/recommendation_model.pkl",
    "recommendation_scaler": "models/recommendation_scaler.pkl"
}

# ==============================
# Initialize Global Registry
# ==============================
registry = ModelRegistry()

def initialize_models():
    """
    Load all models at startup
    """

    for name, path in MODEL_PATHS.items():
        registry.register(name, path)

    print("\n[INFO] All models initialized")

# ==============================
# Quick Access Functions
# ==============================
def get_model(name):
    return registry.get(name)

def reload_model(name):
    if name in MODEL_PATHS:
        registry.reload(name, MODEL_PATHS[name])

def list_all_models():
    return registry.list_models()

# ==============================
# TEST RUN
# ==============================
if __name__ == "__main__":
    initialize_models()
    print("Available models:", list_all_models())