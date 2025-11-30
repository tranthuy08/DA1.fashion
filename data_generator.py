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
    
    Parameters:
    - df: pd.DataFrame
    - weights: dict, keys are numeric feature names, values are weights
    - threshold_quantile: float, quantile to define trending threshold

    Returns:
    - df: pd.DataFrame with 'trend_score' and 'Trend' columns
    """
    if weights is None:
        weights = config.TREND_SCORE_WEIGHTS
    if threshold_quantile is None:
        threshold_quantile = config.TREND_THRESHOLD_QUANTILE

    # Ensure numeric columns exist
    for k in weights.keys():
        if k not in df.columns:
            df[k] = 0

    # Scale numeric columns 0-1
    scaler = MinMaxScaler()
    numeric_cols = list(weights.keys())
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Compute trend_score using weights
    score = sum(df[k] * w for k, w in weights.items())
    df['trend_score'] = score

    # Determine threshold and assign Trend label
    threshold = score.quantile(threshold_quantile)
    df['Trend'] = (score >= threshold).astype(int)

    return df

def generate_fashion_data():
    """
    Load and preprocess fashion data, returning dataframe with Trend column.
    """
    df = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")

    # Fill missing numeric columns
    if 'rating' in df.columns:
        df['rating'] = df['rating'].fillna(df['rating'].median())
    for col in [c for c in config.NUMERIC_FEATURES if c != 'rating']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill text columns
    for col in config.TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Fill categorical columns
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")

    # Add Trend label
    df = add_trend_label(df)

    return df

def get_data_summary(df):
    """
    Generate summary statistics from the dataframe.
    Returns numeric values for avg_price and avg_rating.
    """
    trending_df = df[df['Trend'] == 1]
    summary = {
        'total_items': len(df),
        'trending_items': len(trending_df),
        'trending_ratio': len(trending_df) / len(df) if len(df) > 0 else 0,
        'unique_colors': int(df['color'].nunique()) if 'color' in df.columns else 0,
        'unique_styles': int(df['style attributes'].nunique()) if 'style attributes' in df.columns else 0,
        'unique_brands': int(df['brand'].nunique()) if 'brand' in df.columns else 0,
        'avg_price': float(df['price'].mean()) if 'price' in df.columns else None,
        'avg_rating': float(df['rating'].mean()) if 'rating' in df.columns else None,
        'date_range': 'Historical Fashion Data'
    }
    return summary
