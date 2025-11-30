"""
Data loading and summary functions for Fashion Trend Prediction
"""

import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import config

def add_trend_label(df, weights=None, threshold_quantile=None):
    """
    Add 'trend_score' and 'Trend' label based on weighted numeric features.
    Preserves original numeric values for display purposes.

    Parameters:
    - df: pd.DataFrame
    - weights: dict, keys are numeric feature names, values are weights
    - threshold_quantile: float, quantile to define trending threshold

    Returns:
    - df: pd.DataFrame with 'trend_score', 'Trend' and '_orig' columns
    """
    df = df.copy()
    if weights is None:
        weights = config.TREND_SCORE_WEIGHTS
    if threshold_quantile is None:
        threshold_quantile = config.TREND_THRESHOLD_QUANTILE

    # Preserve original numeric columns for avg calculation
    for col in weights.keys():
        if col in df.columns:
            df[f'{col}_orig'] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Ensure numeric columns exist
    for col in weights.keys():
        if col not in df.columns:
            df[col] = 0

    numeric_cols = list(weights.keys())
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Compute trend_score
    df['trend_score'] = sum(df[k] * w for k, w in weights.items())

    threshold = df['trend_score'].quantile(threshold_quantile)
    df['Trend'] = (df['trend_score'] >= threshold).astype(int)

    return df

def generate_fashion_data():
    df = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")

    # Fill missing numeric
    for col in config.NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Fill text columns
    for col in config.TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Fill categorical columns
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")

    df = add_trend_label(df)
    df['is_trending'] = df['Trend']

    return df

def get_data_summary(df):
    if df.empty:
        return {
            'total_items': 0,
            'trending_items': 0,
            'trending_ratio': 0.0,
            'unique_colors': 0,
            'unique_styles': 0,
            'unique_brands': 0,
            'avg_price': None,
            'avg_rating': None,
            'date_range': 'Historical Fashion Data'
        }

    trending_df = df[df['Trend'] == 1]

    summary = {
        'total_items': len(df),
        'trending_items': len(trending_df),
        'trending_ratio': len(trending_df) / len(df),
        'unique_colors': int(df['color'].nunique()),
        'unique_styles': int(df['style attributes'].nunique()),
        'unique_brands': int(df['brand'].nunique()),
        'avg_price': df['price_orig'].mean() if 'price_orig' in df.columns else None,
        'avg_rating': df['rating_orig'].mean() if 'rating_orig' in df.columns else None,
        'date_range': 'Historical Fashion Data'
    }

    return summary