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

    # Preserve original numeric columns
    for col in weights.keys():
        if col in df.columns:
            df[f'{col}_orig'] = df[col]

    # Scale numeric columns 0-1
    numeric_cols = [col for col in weights.keys() if col in df.columns]
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Compute trend_score using weights
    df['trend_score'] = sum(df[col] * w for col, w in weights.items() if col in df.columns)

    # Determine threshold and assign Trend label
    threshold = df['trend_score'].quantile(threshold_quantile)
    df['Trend'] = (df['trend_score'] >= threshold).astype(int)

    return df


def generate_fashion_data():
    """
    Load and preprocess fashion data, returning dataframe with Trend column.
    Preserves original numeric columns for EDA.
    """
    # Load CSV
    df = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")

    # Add Trend label (with original numeric columns preserved)
    df = add_trend_label(df)

    # Add helper column for Streamlit logic
    df['is_trending'] = df['Trend']

    return df


def get_data_summary(df):
    """
    Generate summary statistics from the dataframe.
    Returns numeric values for avg_price, avg_rating, etc., safely.
    """
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

    def safe_mean(series):
        if series is not None and not series.dropna().empty:
            return float(series.mean())
        return None

    summary = {
        'total_items': len(df),
        'trending_items': len(trending_df),
        'trending_ratio': len(trending_df) / len(df) if len(df) > 0 else 0.0,
        'unique_colors': int(df['color'].nunique()) if 'color' in df.columns else 0,
        'unique_styles': int(df['style attributes'].nunique()) if 'style attributes' in df.columns else 0,
        'unique_brands': int(df['brand'].nunique()) if 'brand' in df.columns else 0,
        'avg_price': safe_mean(df['price']) if 'price' in df.columns else None,
        'avg_rating': safe_mean(df['rating']) if 'rating' in df.columns else None,
        'date_range': 'Historical Fashion Data'
    }

    return summary
