# ==========================================
# utils/analytics_engine.py
# AI Driven Customer Analytics Platform
# Advanced Analytics Engine
# ==========================================

import pandas as pd
import numpy as np


# ==========================================
# KPI CALCULATIONS
# ==========================================
def calculate_kpis(df):

    if df is None or len(df) == 0:
        return {
            "total_customers": 0,
            "total_revenue": 0,
            "avg_purchase": 0,
            "avg_income": 0,
            "high_value_customers": 0,
            "churn_rate": 0,
            "avg_engagement": 0,
            "avg_purchase_probability": 0
        }

    total_customers = len(df)

    total_revenue = float(
        df["purchase_amount"].sum()
    )

    avg_purchase = float(
        df["purchase_amount"].mean()
    )

    avg_income = float(
        df["income"].mean()
    )

    high_value_customers = int(
        df["high_value_customer"].sum()
    )

    churn_rate = float(
        df["churn_probability"].mean() * 100
    )

    avg_engagement = float(
        df["engagement_score"].mean()
    )

    avg_purchase_probability = float(
        df["purchase_probability"].mean() * 100
    )

    return {
        "total_customers": total_customers,
        "total_revenue": round(total_revenue, 2),
        "avg_purchase": round(avg_purchase, 2),
        "avg_income": round(avg_income, 2),
        "high_value_customers": high_value_customers,
        "churn_rate": round(churn_rate, 2),
        "avg_engagement": round(avg_engagement, 2),
        "avg_purchase_probability": round(
            avg_purchase_probability, 2
        )
    }


# ==========================================
# REVENUE ANALYTICS
# ==========================================
def revenue_analytics(df):

    return {
        "total_revenue":
            float(df["purchase_amount"].sum()),

        "average_revenue":
            float(df["purchase_amount"].mean()),

        "max_purchase":
            float(df["purchase_amount"].max()),

        "min_purchase":
            float(df["purchase_amount"].min())
    }


# ==========================================
# CUSTOMER SEGMENT ANALYTICS
# ==========================================
def segment_analytics(df):

    result = []

    grouped = df.groupby("segment")

    for segment, data in grouped:

        result.append({
            "segment": int(segment),

            "customers":
                int(len(data)),

            "avg_income":
                round(
                    float(
                        data["income"].mean()
                    ), 2
                ),

            "avg_purchase":
                round(
                    float(
                        data["purchase_amount"].mean()
                    ), 2
                ),

            "avg_churn":
                round(
                    float(
                        data["churn_probability"].mean()
                    ) * 100, 2
                )
        })

    return result


# ==========================================
# CHURN ANALYTICS
# ==========================================
def churn_analytics(df):

    high_risk = int(
        (
            df["churn_probability"] > 0.7
        ).sum()
    )

    medium_risk = int(
        (
            (df["churn_probability"] > 0.4) &
            (df["churn_probability"] <= 0.7)
        ).sum()
    )

    low_risk = int(
        (
            df["churn_probability"] <= 0.4
        ).sum()
    )

    return {
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
        "average_churn":
            round(
                float(
                    df["churn_probability"].mean()
                ) * 100, 2
            )
    }


# ==========================================
# HIGH VALUE CUSTOMERS
# ==========================================
def top_customers(df, top_n=10):

    data = (
        df.sort_values(
            "purchase_amount",
            ascending=False
        )
        .head(top_n)
        [
            [
                "customer_id",
                "purchase_amount",
                "income",
                "segment"
            ]
        ]
    )

    return data.to_dict(
        orient="records"
    )


# ==========================================
# CUSTOMER LIFETIME VALUE
# ==========================================
def calculate_clv(df):

    df = df.copy()

    df["customer_lifetime_value"] = (
        df["purchase_amount"] *
        df["engagement_score"]
    )

    return df


# ==========================================
# CATEGORY ANALYTICS
# ==========================================
def category_analytics(df):

    category_count = (
        df["preferred_category"]
        .value_counts()
        .reset_index()
    )

    category_count.columns = [
        "category",
        "customers"
    ]

    return category_count.to_dict(
        orient="records"
    )


# ==========================================
# AGE GROUP ANALYTICS
# ==========================================
def age_group_analytics(df):

    bins = [
        18,
        25,
        35,
        45,
        55,
        100
    ]

    labels = [
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56+"
    ]

    df = df.copy()

    df["age_group"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    age_summary = (
        df["age_group"]
        .value_counts()
        .reset_index()
    )

    age_summary.columns = [
        "age_group",
        "customers"
    ]

    return age_summary.to_dict(
        orient="records"
    )


# ==========================================
# CORRELATION MATRIX
# ==========================================
def correlation_matrix(df):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()

    return corr.round(2).to_dict()


# ==========================================
# DATASET SUMMARY
# ==========================================
def dataset_summary(df):

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values":
            int(df.isnull().sum().sum()),

        "duplicate_rows":
            int(df.duplicated().sum())
    }


# ==========================================
# EXECUTIVE DASHBOARD PACKAGE
# ==========================================
def dashboard_package(df):

    return {
        "kpis":
            calculate_kpis(df),

        "revenue":
            revenue_analytics(df),

        "segments":
            segment_analytics(df),

        "churn":
            churn_analytics(df),

        "top_customers":
            top_customers(df),

        "categories":
            category_analytics(df),

        "age_groups":
            age_group_analytics(df),

        "dataset":
            dataset_summary(df)
    }