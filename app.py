import os
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib

import matplotlib.pyplot as plt

# If joblib needs xgboost symbols during load
try:
    import xgboost  # noqa: F401
except ImportError:
    pass

# ------------------------------------------------------------
# Paths (adjust if your structure is different)
# ------------------------------------------------------------
POSSIBLE_MODEL_PATHS = [
    "models/final_model_xgboost.pkl",
    "../models/final_model_xgboost.pkl",
]

POSSIBLE_DATA_PATHS = [
    "data/processed/cleaned_listings.csv",
    "../data/processed/cleaned_listings.csv",
]

# ------------------------------------------------------------
# Helper: find first existing path from a list
# ------------------------------------------------------------
def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


MODEL_PATH = first_existing(POSSIBLE_MODEL_PATHS)
DATA_PATH = first_existing(POSSIBLE_DATA_PATHS)

# ------------------------------------------------------------
# Feature engineering – SAME as in Notebook 2 / 3
# ------------------------------------------------------------
FEATURE_COLS = [
    "product_weight_g",
    "product_volume_cm3",
    "freight_value",
    "category_median_price",
    "category_mean_price",
    "category_price_std",
    "category_count",
    "seller_median_price",
    "seller_avg_freight",
    "seller_total_items",
    "purchase_month",
    "purchase_dayofweek",
]
TARGET_COL = "price"

UNDER_THRESHOLD = 0.20   # underpriced if delta_pct > 20%
OVER_THRESHOLD = -0.20   # overpriced if delta_pct < -20%
MAX_UP = 0.30            # cap upward change (+30%)
MAX_DOWN = 0.05          # cap downward change (-5%)


def build_model_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Recreate the modeling dataset exactly as in Notebook 2/3.

    Returns:
        listings_model : dataframe with all engineered features
        model_df       : subset with FEATURE_COLS + price (no missing)
        X              : feature matrix
        y              : target (price)
    """
    df = df_raw.copy()

    # Sanity checks
    assert "category_english" in df.columns, "category_english column missing."
    assert "product_volume_cm3" in df.columns, "product_volume_cm3 missing."
    assert "price" in df.columns, "price column (target) missing."

    # 1) Category-level aggregations
    category_stats = df.groupby("category_english").agg(
        category_median_price=("price", "median"),
        category_mean_price=("price", "mean"),
        category_price_std=("price", "std"),
        category_count=("price", "count"),
    ).reset_index()

    df = df.merge(category_stats, on="category_english", how="left")

    # 2) Seller-level features
    seller_stats = df.groupby("seller_id").agg(
        seller_median_price=("price", "median"),
        seller_avg_freight=("freight_value", "mean"),
        seller_total_items=("order_id", "count"),
    ).reset_index()

    df = df.merge(seller_stats, on="seller_id", how="left")

    # 3) Time features
    purchase_dt = pd.to_datetime(df["order_purchase_timestamp"])
    df["purchase_month"] = purchase_dt.dt.month
    df["purchase_dayofweek"] = purchase_dt.dt.dayofweek

    # 4) Final feature set
    model_df = df[FEATURE_COLS + [TARGET_COL]].dropna()
    X = model_df[FEATURE_COLS]
    y = model_df[TARGET_COL]

    return df, model_df, X, y


def classify_pricing_flag(delta_pct: float) -> str:
    if delta_pct > UNDER_THRESHOLD:
        return "underpriced"
    elif delta_pct < OVER_THRESHOLD:
        return "overpriced"
    else:
        return "fair"


def apply_pricing_rules(row):
    actual = row["actual_price"]
    pred = row["predicted_price"]
    flag = row["pricing_flag"]

    # Move partially toward model price
    halfway_price = 0.7 * actual + 0.3 * pred
    raw_adj_pct = (halfway_price - actual) / actual

    if flag == "underpriced":
        # upward adjustment, capped
        adj_pct = min(max(raw_adj_pct, 0), MAX_UP)
    elif flag == "overpriced":
        # downward adjustment, capped
        adj_pct = max(min(raw_adj_pct, 0), -MAX_DOWN)
    else:
        adj_pct = 0.0

    return actual * (1 + adj_pct)


# ------------------------------------------------------------
# Load data + model (cached)
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_base_data(path):
    return pd.read_csv(path)


@st.cache_resource(show_spinner=True)
def load_model(path):
    return joblib.load(path)


@st.cache_data(show_spinner=True)
def prepare_full_simulation(model_path, data_path):
    """
    Full pipeline:
      - load data
      - build features
      - predict prices
      - classify mispricing
      - apply pricing rules
      - compute revenue metrics
    """
    # Load raw cleaned listings
    df_raw = load_base_data(data_path)

    # Build features as in Notebook 2/3
    listings_model, model_df, X, y = build_model_dataset(df_raw)

    # Load trained model
    final_model = load_model(model_path)

    # Predict
    predicted = final_model.predict(X)

    model_df = model_df.copy()
    model_df["actual_price"] = y
    model_df["predicted_price"] = predicted
    model_df["price_delta"] = model_df["predicted_price"] - model_df["actual_price"]
    model_df["delta_pct"] = model_df["price_delta"] / model_df["actual_price"]

    # Mispricing flags
    model_df["pricing_flag"] = model_df["delta_pct"].apply(classify_pricing_flag)

    # Recommended prices
    model_df["recommended_price"] = model_df.apply(apply_pricing_rules, axis=1)

    # Revenues
    model_df["revenue_actual"] = model_df["actual_price"]
    model_df["revenue_recommended"] = model_df["recommended_price"]

    # Attach category_english for each row (align indices)
    sim_df = listings_model.loc[model_df.index, ["category_english"]].copy()
    sim_df["revenue_actual"] = model_df["revenue_actual"].values
    sim_df["revenue_recommended"] = model_df["revenue_recommended"].values

    # Revenue by category
    revenue_by_cat = (
        sim_df
        .groupby("category_english")[["revenue_actual", "revenue_recommended"]]
        .sum()
        .sort_values("revenue_actual", ascending=False)
    )

    revenue_by_cat["uplift_abs"] = (
        revenue_by_cat["revenue_recommended"] - revenue_by_cat["revenue_actual"]
    )
    revenue_by_cat["uplift_pct"] = (
        revenue_by_cat["uplift_abs"] / revenue_by_cat["revenue_actual"] * 100
    )

    # Revenue by pricing flag
    revenue_by_flag = (
        model_df
        .groupby("pricing_flag")[["revenue_actual", "revenue_recommended"]]
        .sum()
        .sort_values("revenue_actual", ascending=False)
    )

    revenue_by_flag["uplift_abs"] = (
        revenue_by_flag["revenue_recommended"] - revenue_by_flag["revenue_actual"]
    )
    revenue_by_flag["uplift_pct"] = (
        revenue_by_flag["uplift_abs"] / revenue_by_flag["revenue_actual"] * 100
    )

    total_actual_rev = revenue_by_flag["revenue_actual"].sum()
    revenue_by_flag["share_of_total_actual"] = (
        revenue_by_flag["revenue_actual"] / total_actual_rev * 100
    )

    # Overall KPIs
    total_actual = model_df["revenue_actual"].sum()
    total_reco = model_df["revenue_recommended"].sum()
    uplift_abs = total_reco - total_actual
    uplift_pct = uplift_abs / total_actual * 100

    kpis = {
        "total_actual": total_actual,
        "total_reco": total_reco,
        "uplift_abs": uplift_abs,
        "uplift_pct": uplift_pct,
    }

    return model_df, revenue_by_cat, revenue_by_flag, kpis


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic Pricing Strategy Demo",
    layout="wide"
)

st.title("🧮 Dynamic Pricing Strategy – Streamlit Demo")

# Check paths
if MODEL_PATH is None or DATA_PATH is None:
    st.error(
        "Could not locate model or data.\n\n"
        f"MODEL_PATH candidates: {POSSIBLE_MODEL_PATHS}\n"
        f"DATA_PATH candidates: {POSSIBLE_DATA_PATHS}"
    )
    st.stop()

st.sidebar.success(f"Model: {MODEL_PATH}")
st.sidebar.success(f"Data:  {DATA_PATH}")

# Prepare everything
with st.spinner("Running pricing simulation..."):
    model_df, revenue_by_cat, revenue_by_flag, kpis = prepare_full_simulation(
        MODEL_PATH,
        DATA_PATH,
    )

# ------------------------------------------------------------
# TOP KPIs
# ------------------------------------------------------------
st.subheader("Overall Revenue Impact")

col1, col2, col3 = st.columns(3)
col1.metric(
    "Total Actual Revenue",
    f"${kpis['total_actual']:,.0f}"
)
col2.metric(
    "Total Recommended Revenue",
    f"${kpis['total_reco']:,.0f}"
)
col3.metric(
    "Revenue Uplift",
    f"${kpis['uplift_abs']:,.0f}",
    f"{kpis['uplift_pct']:.2f}%"
)

st.caption(
    "Each row represents a sold item. We recompute a fair model price, "
    "flag under/overpricing, and apply conservative pricing rules."
)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📦 By Category", "🔎 Row Explorer"])

# ============================================================
# TAB 1 – Overview
# ============================================================
with tab1:
    st.markdown("### Pricing Segments & Revenue")

    st.write("**Revenue by pricing segment (actual vs recommended):**")
    st.dataframe(
        revenue_by_flag.style.format({
            "revenue_actual": "{:,.2f}",
            "revenue_recommended": "{:,.2f}",
            "uplift_abs": "{:,.2f}",
            "uplift_pct": "{:,.2f}",
            "share_of_total_actual": "{:,.2f}",
        }),
        use_container_width=True,
    )

    st.markdown("#### Revenue by pricing segment (chart)")
    fig, ax = plt.subplots(figsize=(6, 4))
    revenue_by_flag[["revenue_actual", "revenue_recommended"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Revenue")
    ax.set_xlabel("Pricing segment")
    ax.set_title("Actual vs Recommended Revenue by Pricing Segment")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    st.markdown("#### Distribution of pricing flags (rows)")
    flag_counts = model_df["pricing_flag"].value_counts(normalize=True).sort_index()
    flag_counts_df = (flag_counts * 100).rename("share_pct").to_frame()
    st.dataframe(
        flag_counts_df.style.format({"share_pct": "{:,.2f}"}),
        use_container_width=True,
    )

# ============================================================
# TAB 2 – By Category
# ============================================================
with tab2:
    st.markdown("### Category-Level Revenue Impact")

    st.write("Top categories by actual revenue:")
    st.dataframe(
        revenue_by_cat.head(20).style.format({
            "revenue_actual": "{:,.2f}",
            "revenue_recommended": "{:,.2f}",
            "uplift_abs": "{:,.2f}",
            "uplift_pct": "{:,.2f}",
        }),
        use_container_width=True,
    )

    st.markdown("#### Actual vs Recommended Revenue by Category (Top 10)")
    top_n = st.slider("Number of top categories to plot", min_value=5, max_value=20, value=10, step=1)
    top_cat = revenue_by_cat.head(top_n)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    top_cat[["revenue_actual", "revenue_recommended"]].plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Revenue")
    ax2.set_xlabel("Category")
    ax2.set_title("Actual vs Recommended Revenue by Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)

# ============================================================
# TAB 3 – Row Explorer
# ============================================================
with tab3:
    st.markdown("### Row-Level Pricing Explorer")

    # Bring in category + maybe other context from original DF
    # (We know model_df indices align with listings_model indices from earlier)
    # For this app we only have model_df here, so we keep it numeric + flags.
    # To make exploration easier, add a simple row_id
    df_explorer = model_df.copy()
    df_explorer = df_explorer.reset_index(drop=True)
    df_explorer["row_id"] = df_explorer.index

    # Filters
    colA, colB = st.columns(2)
    with colA:
        flag_filter = st.multiselect(
            "Pricing segment",
            options=sorted(df_explorer["pricing_flag"].unique()),
            default=sorted(df_explorer["pricing_flag"].unique()),
        )
    with colB:
        price_min, price_max = st.slider(
            "Filter by actual price",
            float(df_explorer["actual_price"].min()),
            float(df_explorer["actual_price"].max()),
            (float(df_explorer["actual_price"].min()), float(df_explorer["actual_price"].max())),
        )

    filtered = df_explorer[
        (df_explorer["pricing_flag"].isin(flag_filter))
        & (df_explorer["actual_price"] >= price_min)
        & (df_explorer["actual_price"] <= price_max)
    ]

    st.write(f"Showing {len(filtered):,} rows after filters.")

    # Show a sample to keep it light
    sample_size = st.number_input(
        "Number of rows to preview",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    st.dataframe(
        filtered[
            [
                "row_id",
                "actual_price",
                "predicted_price",
                "recommended_price",
                "pricing_flag",
                "price_delta",
                "delta_pct",
            ]
        ]
        .head(int(sample_size))
        .style.format({
            "actual_price": "{:,.2f}",
            "predicted_price": "{:,.2f}",
            "recommended_price": "{:,.2f}",
            "price_delta": "{:,.2f}",
            "delta_pct": "{:,.2%}",
        }),
        use_container_width=True,
    )

    st.markdown("#### Inspect a single row")
    selected_id = st.number_input(
        "Enter row_id to inspect",
        min_value=0,
        max_value=int(df_explorer["row_id"].max()),
        value=0,
        step=1,
    )

    row = df_explorer.loc[df_explorer["row_id"] == selected_id].iloc[0]

    st.write("**Selected row details:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual price", f"${row['actual_price']:,.2f}")
    col2.metric("Model fair price", f"${row['predicted_price']:,.2f}")
    col3.metric("Recommended price", f"${row['recommended_price']:,.2f}")

    st.write(f"Pricing flag: **{row['pricing_flag']}**")
    st.write(f"Delta vs actual: ${row['price_delta']:,.2f} ({row['delta_pct']:.2%})")