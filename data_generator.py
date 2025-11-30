"""
Data loading and summary functions for Fashion Trend Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import config


def add_trend_label(df, weights=None, threshold_quantile=None):
    """
    Add 'trend_score' and 'Trend' label WITHOUT modifying the original numeric columns.
    """
    if weights is None:
        weights = config.TREND_SCORE_WEIGHTS
    if threshold_quantile is None:
        threshold_quantile = config.TREND_THRESHOLD_QUANTILE

    # Ensure numeric columns exist
    for k in weights.keys():
        if k not in df.columns:
            df[k] = 0

    numeric_cols = list(weights.keys())

    # ---- FIX: scale on a copy, do NOT overwrite df ----
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols)

    # Compute weighted score from scaled copy
    score = np.zeros(len(df))
    for col, w in weights.items():
        score += scaled_df[col] * w

    df["trend_score"] = score
    threshold = np.quantile(score, threshold_quantile)
    df["Trend"] = (score >= threshold).astype(int)

    return df


def generate_fashion_data():
    """
    Load and preprocess fashion data, returning dataframe with Trend column.
    """
    df = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")

    # Fill numeric
    if "rating" in df.columns:
        df["rating"] = df["rating"].fillna(df["rating"].median())
    for col in [c for c in config.NUMERIC_FEATURES if c != "rating"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill text
    for col in config.TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Fill categories
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")

    # Add Trend label
    df = add_trend_label(df)

    return df


def get_data_summary(df):
    """
    Generate summary statistics.
    """
    trending_df = df[df["Trend"] == 1]

    summary = {
        "total_items": len(df),
        "trending_items": len(trending_df),
        "trending_ratio": f"{len(trending_df) / len(df):.1%}" if len(df) else "0%",
        "unique_colors": df["color"].nunique() if "color" in df.columns else 0,
        "unique_styles": df["style attributes"].nunique() if "style attributes" in df.columns else 0,
        "unique_brands": df["brand"].nunique() if "brand" in df.columns else 0,
        "avg_price": f"${df['price'].mean():.2f}" if "price" in df.columns else "N/A",
        "avg_rating": f"{df['rating'].mean():.2f}" if "rating" in df.columns else "N/A",
        "date_range": "Historical Fashion Data",
    }

    return summary
