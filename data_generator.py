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

    # Ensure numeric columns exist
    for col in weights.keys():
        if col not in df.columns:
            df[col] = 0

    # Scale numeric columns 0-1
    numeric_cols = list(weights.keys())
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Compute trend_score using weights
    df['trend_score'] = sum(df[k] * w for k, w in weights.items())

    # Determine threshold and assign Trend label
    threshold = df['trend_score'].quantile(threshold_quantile)
    df['Trend'] = (df['trend_score'] >= threshold).astype(int)

    return df


def generate_fashion_data(n=210000, seed=42):
    """
    Generate mock fashion dataset with consistent numeric/categorical fields.
    """
    np.random.seed(seed)
    random.seed(seed)

    df = pd.DataFrame({
        'price': np.random.uniform(10, 100, n).round(2),
        'rating': np.random.uniform(1, 5, n).round(1),
        'reviews_count': np.random.randint(0, 500, n),
        'age': np.random.randint(15, 65, n),
        'total_sizes': np.random.randint(1, 10, n),
        'Trend': np.random.choice([0,1], size=n, p=[0.6,0.4]),  # realistic trending ratio
        'brand': np.random.choice(config.BRANDS, n),
        'category': np.random.choice(config.CATEGORIES, n),
        'color': np.random.choice(config.COLORS, n),
        'style attributes': np.random.choice(config.STYLE_ATTRIBUTES, n),
        'season': np.random.choice(config.SEASONS, n),
        'size': np.random.choice(config.SIZES, n),
        'description': np.random.choice(config.DESCRIPTIONS, n),
        'purchase_history': np.random.choice(config.PURCHASE_HISTORY, n),
        'fashion_magazines': np.random.choice(config.FASHION_MAGAZINES, n),
        'fashion_influencers': np.random.choice(config.FASHION_INFLUENCERS, n),
        'time_period_highest_purchase': np.random.choice(config.TIME_PERIODS, n),
        'customer_reviews': np.random.choice(config.CUSTOMER_REVIEWS, n),
        'social_media_comments': np.random.choice(config.SOCIAL_COMMENTS, n),
        'feedback': np.random.choice(config.FEEDBACK_OPTIONS, n)
    })
    return df

def get_data_summary(df: pd.DataFrame):
    """
    Return dictionary with summary stats for Overview page.
    """
    total_items = len(df)
    trending_items = df['Trend'].sum()
    trending_ratio = trending_items / total_items if total_items > 0 else 0

    avg_price = df['price'].mean() if 'price' in df.columns else None
    avg_rating = df['rating'].mean() if 'rating' in df.columns else None

    summary = {
        'total_items': total_items,
        'trending_items': trending_items,
        'trending_ratio': trending_ratio,
        'unique_colors': df['color'].nunique() if 'color' in df.columns else 0,
        'unique_styles': df['style attributes'].nunique() if 'style attributes' in df.columns else 0,
        'unique_brands': df['brand'].nunique() if 'brand' in df.columns else 0,
        'avg_price': avg_price,
        'avg_rating': avg_rating
    }
    return summary