# ==============================
# 11_utils/visualization.py
# Advanced Visualization Engine (Plotly)
# ==============================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==============================
# KPI Overview Chart
# ==============================
def kpi_bar_chart(kpis):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(kpis.keys()),
        y=list(kpis.values()),
        marker_color=["blue", "green", "orange", "red"]
    ))

    fig.update_layout(
        title="📊 KPI Overview",
        xaxis_title="Metrics",
        yaxis_title="Values"
    )

    return fig.to_html(full_html=False)

# ==============================
# Customer Segmentation Plot
# ==============================
def segmentation_scatter(df):
    if "pca_x" not in df.columns or "pca_y" not in df.columns:
        return "<p>No PCA data available</p>"

    fig = px.scatter(
        df,
        x="pca_x",
        y="pca_y",
        color="segment",
        title="🎯 Customer Segmentation (PCA View)",
        hover_data=["customer_id"] if "customer_id" in df.columns else None
    )

    return fig.to_html(full_html=False)

# ==============================
# Segment Distribution Pie Chart
# ==============================
def segment_pie_chart(df):
    if "segment" not in df.columns:
        return "<p>No segment data available</p>"

    segment_counts = df["segment"].value_counts().reset_index()
    segment_counts.columns = ["segment", "count"]

    fig = px.pie(
        segment_counts,
        names="segment",
        values="count",
        title="📦 Customer Segment Distribution"
    )

    return fig.to_html(full_html=False)

# ==============================
# Churn Distribution Chart
# ==============================
def churn_pie_chart(df):
    if "churn" not in df.columns:
        return "<p>No churn data available</p>"

    churn_counts = df["churn"].value_counts().reset_index()
    churn_counts.columns = ["churn", "count"]

    fig = px.pie(
        churn_counts,
        names="churn",
        values="count",
        title="⚠️ Churn Distribution"
    )

    return fig.to_html(full_html=False)

# ==============================
# Churn Risk Histogram
# ==============================
def churn_risk_histogram(df):
    if "churn_probability" not in df.columns:
        return "<p>No churn probability data</p>"

    fig = px.histogram(
        df,
        x="churn_probability",
        nbins=20,
        title="📉 Churn Risk Distribution"
    )

    return fig.to_html(full_html=False)

# ==============================
# Purchase Prediction Distribution
# ==============================
def prediction_distribution(df):
    if "purchase_probability" not in df.columns:
        return "<p>No prediction data</p>"

    fig = px.histogram(
        df,
        x="purchase_probability",
        nbins=20,
        title="🛒 Purchase Probability Distribution"
    )

    return fig.to_html(full_html=False)

# ==============================
# Revenue Trend Chart
# ==============================
def revenue_trend(df):
    if "purchase_amount" not in df.columns:
        return "<p>No revenue data available</p>"

    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        trend = df.groupby("date")["purchase_amount"].sum().reset_index()

        fig = px.line(
            trend,
            x="date",
            y="purchase_amount",
            title="📈 Revenue Trend Over Time"
        )

        return fig.to_html(full_html=False)

    return "<p>No date column for trend analysis</p>"

# ==============================
# Feature Importance Chart
# ==============================
def feature_importance_chart(df, feature_col="feature", value_col="importance"):
    if feature_col not in df.columns:
        return "<p>No feature importance data</p>"

    fig = px.bar(
        df,
        x=feature_col,
        y=value_col,
        title="🔍 Feature Importance"
    )

    return fig.to_html(full_html=False)

# ==============================
# Dataset Overview Table
# ==============================
def dataset_table(df):
    return df.head(20).to_html(classes="table table-striped", index=False)

# ==============================
# Correlation Heatmap
# ==============================
def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        title="🔥 Feature Correlation Heatmap"
    )

    return fig.to_html(full_html=False)

# ==============================
# SEGMENT INSIGHT BAR CHART
# ==============================
def segment_insight_chart(summary_df):
    if summary_df.empty:
        return "<p>No segment summary available</p>"

    fig = px.bar(
        summary_df.reset_index(),
        x="segment",
        y="avg_purchase" if "avg_purchase" in summary_df.columns else summary_df.columns[1],
        title="📊 Segment Insights"
    )

    return fig.to_html(full_html=False)