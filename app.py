# ==========================================
# app.py
# AI Driven Customer Analytics Platform
# Complete Flask Backend
# ==========================================

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import os

from generate_dataset import save_dataset

from utils.analytics_engine import (
    calculate_kpis,
    dashboard_package,
    revenue_analytics,
    segment_analytics,
    churn_analytics,
    top_customers,
    category_analytics,
    age_group_analytics,
    correlation_matrix,
    dataset_summary
)

# ==========================================
# FLASK APP
# ==========================================
app = Flask(__name__)
CORS(app)

DATA_PATH = "data/processed_data.csv"


# ==========================================
# LOAD DATASET
# ==========================================
def load_dataset():

    if not os.path.exists(DATA_PATH):
        save_dataset()

    return pd.read_csv(DATA_PATH)


# ==========================================
# GLOBAL DATAFRAME
# ==========================================
df = load_dataset()


# ==========================================
# REFRESH DATASET
# ==========================================
def refresh_data():
    global df
    df = load_dataset()


# ==========================================
# WEB ROUTES
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/segmentation")
def segmentation():
    return render_template("segmentation.html")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/recommendation")
def recommendation():
    return render_template("recommendation.html")


@app.route("/churn")
def churn():
    return render_template("churn.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset_overview.html")


# ==========================================
# DATASET GENERATION
# ==========================================
@app.route("/api/generate_dataset")
def generate_dataset_api():

    save_dataset()
    refresh_data()

    return jsonify({
        "status": "success",
        "message": "Dataset generated successfully"
    })


# ==========================================
# DATASET API
# ==========================================
@app.route("/api/dataset")
def dataset_api():

    refresh_data()

    return jsonify(
        df.to_dict(
            orient="records"
        )
    )


# ==========================================
# KPI API
# ==========================================
@app.route("/api/kpis")
def kpis_api():

    refresh_data()

    return jsonify(
        calculate_kpis(df)
    )


# ==========================================
# DASHBOARD PACKAGE
# ==========================================
@app.route("/api/dashboard")
def dashboard_api():

    refresh_data()

    return jsonify(
        dashboard_package(df)
    )


# ==========================================
# REVENUE ANALYTICS
# ==========================================
@app.route("/api/revenue")
def revenue_api():

    refresh_data()

    return jsonify(
        revenue_analytics(df)
    )


# ==========================================
# SEGMENT ANALYTICS
# ==========================================
@app.route("/api/segments")
def segment_api():

    refresh_data()

    return jsonify(
        segment_analytics(df)
    )


# ==========================================
# CHURN ANALYTICS
# ==========================================
@app.route("/api/churn")
def churn_api():

    refresh_data()

    return jsonify(
        churn_analytics(df)
    )


# ==========================================
# TOP CUSTOMERS
# ==========================================
@app.route("/api/top_customers")
def top_customers_api():

    refresh_data()

    return jsonify(
        top_customers(df)
    )


# ==========================================
# CATEGORY ANALYTICS
# ==========================================
@app.route("/api/categories")
def categories_api():

    refresh_data()

    return jsonify(
        category_analytics(df)
    )


# ==========================================
# AGE GROUP ANALYTICS
# ==========================================
@app.route("/api/age_groups")
def age_groups_api():

    refresh_data()

    return jsonify(
        age_group_analytics(df)
    )


# ==========================================
# CORRELATION MATRIX
# ==========================================
@app.route("/api/correlation")
def correlation_api():

    refresh_data()

    return jsonify(
        correlation_matrix(df)
    )


# ==========================================
# DATASET SUMMARY
# ==========================================
@app.route("/api/dataset_summary")
def dataset_summary_api():

    refresh_data()

    return jsonify(
        dataset_summary(df)
    )


# ==========================================
# SEGMENTATION DATA
# ==========================================
@app.route("/api/segmentation")
def segmentation_api():

    refresh_data()

    columns = [
        "customer_id",
        "segment",
        "pca_x",
        "pca_y",
        "income",
        "purchase_amount"
    ]

    return jsonify(
        df[columns].to_dict(
            orient="records"
        )
    )


# ==========================================
# PURCHASE PREDICTION
# ==========================================
@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict_api():

    data = request.get_json()

    income = float(
        data.get("income", 0)
    )

    purchase_amount = float(
        data.get("purchase_amount", 0)
    )

    website_visits = float(
        data.get("website_visits", 0)
    )

    score = (
        (income / 150000) * 0.4 +
        (purchase_amount / 5000) * 0.3 +
        (website_visits / 50) * 0.3
    )

    score = round(
        min(score, 1.0),
        4
    )

    prediction = (
        "Likely To Purchase"
        if score > 0.5
        else
        "Low Purchase Probability"
    )

    return jsonify({
        "prediction": prediction,
        "probability": score
    })


# ==========================================
# RECOMMENDATION SYSTEM
# ==========================================
@app.route(
    "/api/recommend",
    methods=["POST"]
)
def recommendation_api():

    data = request.get_json()

    customer_id = int(
        data.get("customer_id", 1)
    )

    refresh_data()

    customer = df[
        df["customer_id"] == customer_id
    ]

    if customer.empty:

        return jsonify({
            "error":
            "Customer not found"
        })

    category = (
        customer.iloc[0]
        ["preferred_category"]
    )

    recommendations = (
        df[
            df["preferred_category"]
            == category
        ]
        .head(5)
        ["customer_id"]
        .tolist()
    )

    return jsonify({

        "customer_id":
            customer_id,

        "preferred_category":
            category,

        "recommended_products": [
            f"{category} Product 1",
            f"{category} Product 2",
            f"{category} Product 3",
            f"{category} Product 4"
        ],

        "similar_customers":
            recommendations
    })


# ==========================================
# HEALTH CHECK
# ==========================================
@app.route("/api/health")
def health():

    return jsonify({
        "status": "running",
        "records": len(df)
    })


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )