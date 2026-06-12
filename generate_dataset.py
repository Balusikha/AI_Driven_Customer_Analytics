# ==========================================
# generate_dataset.py
# AI Driven Customer Analytics
# Dataset Generator (100 Records)
# ==========================================

import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIG
# ==========================================
NUM_RECORDS = 100
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ==========================================
# CUSTOMER CATEGORY GENERATOR
# ==========================================
CATEGORIES = [
    "Electronics",
    "Fashion",
    "Books",
    "Sports",
    "Grocery",
    "Beauty",
    "Home Decor"
]


# ==========================================
# GENERATE DATASET
# ==========================================
def generate_dataset(records=NUM_RECORDS):

    customer_id = np.arange(1, records + 1)

    age = np.random.randint(18, 65, records)

    income = np.random.randint(
        20000,
        150000,
        records
    )

    purchase_amount = np.random.randint(
        100,
        5000,
        records
    )

    last_purchase_days = np.random.randint(
        1,
        365,
        records
    )

    website_visits = np.random.randint(
        1,
        50,
        records
    )

    support_calls = np.random.randint(
        0,
        10,
        records
    )

    preferred_category = np.random.choice(
        CATEGORIES,
        records
    )

    # ======================================
    # CUSTOMER SEGMENT
    # ======================================
    segment = []

    for i in range(records):

        if income[i] > 100000:
            segment.append(2)

        elif income[i] > 60000:
            segment.append(1)

        else:
            segment.append(0)

    segment = np.array(segment)

    # ======================================
    # ENGAGEMENT SCORE
    # ======================================
    engagement_score = (
        website_visits * 0.5
        + support_calls * 1.5
        + purchase_amount / 1000
    )

    # ======================================
    # RECENCY SCORE
    # ======================================
    recency_score = (
        1 /
        (1 + last_purchase_days)
    )

    # ======================================
    # PURCHASE PROBABILITY
    # ======================================
    purchase_probability = (
        (
            engagement_score /
            engagement_score.max()
        ) * 0.6
        +
        (
            income /
            income.max()
        ) * 0.4
    )

    purchase_probability = np.clip(
        purchase_probability,
        0,
        1
    )

    # ======================================
    # CHURN PROBABILITY
    # ======================================
    churn_probability = (
        (
            last_purchase_days /
            last_purchase_days.max()
        ) * 0.6
        +
        (
            1 -
            engagement_score /
            engagement_score.max()
        ) * 0.4
    )

    churn_probability = np.clip(
        churn_probability,
        0,
        1
    )

    # ======================================
    # CHURN LABEL
    # ======================================
    churn = (
        churn_probability > 0.55
    ).astype(int)

    # ======================================
    # HIGH VALUE CUSTOMER
    # ======================================
    high_value_customer = (
        purchase_amount >
        np.percentile(
            purchase_amount,
            75
        )
    ).astype(int)

    # ======================================
    # CUSTOMER LIFETIME VALUE
    # ======================================
    customer_lifetime_value = (
        purchase_amount *
        engagement_score
    )

    # ======================================
    # LOG PURCHASE
    # ======================================
    log_purchase = np.log1p(
        purchase_amount
    )

    # ======================================
    # PCA MOCK VALUES
    # (Used for cluster visualizations)
    # ======================================
    pca_x = np.random.normal(
        segment * 2,
        0.8
    )

    pca_y = np.random.normal(
        segment * 1.5,
        0.8
    )

    # ======================================
    # DATAFRAME
    # ======================================
    df = pd.DataFrame({

        "customer_id":
            customer_id,

        "age":
            age,

        "income":
            income,

        "purchase_amount":
            purchase_amount,

        "last_purchase_days":
            last_purchase_days,

        "website_visits":
            website_visits,

        "support_calls":
            support_calls,

        "segment":
            segment,

        "preferred_category":
            preferred_category,

        "engagement_score":
            engagement_score.round(2),

        "recency_score":
            recency_score.round(4),

        "purchase_probability":
            purchase_probability.round(4),

        "churn_probability":
            churn_probability.round(4),

        "churn":
            churn,

        "high_value_customer":
            high_value_customer,

        "customer_lifetime_value":
            customer_lifetime_value.round(2),

        "log_purchase":
            log_purchase.round(4),

        "pca_x":
            pca_x.round(3),

        "pca_y":
            pca_y.round(3)
    })

    return df


# ==========================================
# SAVE DATASET
# ==========================================
def save_dataset():

    os.makedirs(
        "data",
        exist_ok=True
    )

    df = generate_dataset()

    output_path = (
        "data/processed_data.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n================================")
    print("DATASET GENERATED SUCCESSFULLY")
    print("================================")
    print(
        f"Records : {len(df)}"
    )
    print(
        f"Columns : {len(df.columns)}"
    )
    print(
        f"Saved   : {output_path}"
    )
    print("================================")

    return df


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    save_dataset()