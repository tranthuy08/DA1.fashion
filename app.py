"""
Fashion Trend Prediction System - Streamlit App (Fixed Version)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Custom modules 
from data_generator import generate_fashion_data, get_data_summary
from data_processor import FashionTrendPredictor
import visualization as viz
import config
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Streamlit page config
st.set_page_config(page_title="Fashion Trend Prediction System", page_icon="👗", layout="wide")

# Simple styling
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #FF6B6B; text-align: center; padding: 0.5rem; }
    .sub-header { font-size: 1.25rem; color: #4ECDC4; margin-top: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Load & Train functions
# -----------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = generate_fashion_data()
    # Ensure numeric columns are correct
    for col in ['price','rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_resource(show_spinner=False)
def train_and_get_model(df):
    predictor = FashionTrendPredictor(
        iterations=config.CATBOOST_ITERATIONS,
        learning_rate=config.CATBOOST_LEARNING_RATE,
        depth=config.CATBOOST_DEPTH
    )

    # Prepare data (numeric scaled inside training)
    df_prepared, cat_idx = predictor.prepare_data(df)
    X = df_prepared
    y = df['Trend']

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )

    predictor.train(X_train, y_train, X_val=X_test, y_val=y_test, cat_features_idx=cat_idx)

    test_metrics = predictor.evaluate(X_test, y_test)
    try:
        from sklearn.metrics import roc_auc_score
        y_proba = predictor.model.predict_proba(X_test)[:, 1]
        test_metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    except Exception:
        test_metrics.setdefault('roc_auc', None)

    feature_importance = predictor.get_feature_importance()

    # Save scaler to reuse for prediction
    scaler = MinMaxScaler()
    scaler.fit(df[config.NUMERIC_FEATURES])
    
    return predictor, test_metrics, feature_importance, scaler

# -----------------------
# Main App
# -----------------------
def main():
    st.markdown('<div class="main-header">👗 Fashion Trend Prediction System</div>', unsafe_allow_html=True)
    st.write("Powered by CatBoost — analyze, visualize and predict trending fashion items.")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "📊 Overview",
        "📈 EDA Report",
        "🔍 Data Analysis",
        "🤖 Model Performance",
        "🎨 Trend Insights",
        "🔮 Make Predictions"
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("This dashboard predicts whether an item is *trending* using CatBoost model.")

    with st.spinner("Loading data..."):
        df_raw = load_data()  # gốc, chưa scale

    with st.spinner("Training model (cached)..."):
        predictor, metrics, feature_importance, scaler = train_and_get_model(df_raw)

    if page == "📊 Overview":
        page_overview(df_raw, metrics)
    elif page == "📈 EDA Report":
        page_eda(df_raw)
    elif page == "🔍 Data Analysis":
        page_data_analysis(df_raw)
    elif page == "🤖 Model Performance":
        page_model_performance(metrics, feature_importance)
    elif page == "🎨 Trend Insights":
        page_trend_insights(df_raw, feature_importance)
    elif page == "🔮 Make Predictions":
        page_predictions(predictor, scaler)

# -----------------------
# Pages
# -----------------------
def page_overview(df, metrics):
    st.markdown('<div class="sub-header">📊 System Overview</div>', unsafe_allow_html=True)
    summary = get_data_summary(df)

    # Safe retrieval and formatting
    total_items = summary.get('total_items', 0)
    unique_colors = summary.get('unique_colors', 0)
    trending_items = summary.get('trending_items', 0)
    unique_styles = summary.get('unique_styles', 0)
    trending_ratio = summary.get('trending_ratio', 0)
    unique_brands = summary.get('unique_brands', 0)
    avg_price = summary.get('avg_price', None)
    avg_rating = summary.get('avg_rating', None)

    trending_ratio_str = f"{trending_ratio*100:.2f}%" if trending_ratio is not None else "N/A"
    avg_price_str = f"${avg_price:.2f}" if avg_price is not None else "N/A"
    avg_rating_str = f"{avg_rating:.2f}" if avg_rating is not None else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Items", total_items)
    c1.metric("Unique Colors", unique_colors)
    c2.metric("Trending Items", trending_items)
    c2.metric("Unique Styles", unique_styles)
    c3.metric("Trending Ratio", trending_ratio_str)
    c3.metric("Unique Brands", unique_brands)
    c4.metric("Avg Price", avg_price_str)
    c4.metric("Avg Rating", avg_rating_str)

    st.markdown("---")
    st.markdown("### Model Snapshot")
    cols = st.columns(2)
    acc = metrics.get('accuracy', 0)
    f1 = metrics.get('f1_score', 0)
    roc = metrics.get('roc_auc', None)
    cols[0].write(f"**Accuracy:** {acc:.4f}" if acc is not None else "Accuracy: N/A")
    cols[0].write(f"**F1-score:** {f1:.4f}" if f1 is not None else "F1: N/A")
    cols[1].write(f"**ROC-AUC:** {roc:.4f}" if roc is not None else "ROC-AUC: N/A")

def page_trend_insights(df, feature_importance):
    st.markdown('<div class="sub-header">🎨 Trend Insights</div>', unsafe_allow_html=True)
    trending = df[df['Trend']==1]
    non_trending = df[df['Trend']==0]

    # Top trending attributes
    c1, c2, c3 = st.columns(3)
    c1.write(trending['color'].value_counts().head(10) if not trending.empty else "No data")
    c2.write(trending['style attributes'].value_counts().head(10) if not trending.empty else "No data")
    c3.write(trending['brand'].value_counts().head(10) if not trending.empty else "No data")

    st.markdown("---")
    st.markdown("#### Statistical Summary (Trending vs Non-Trending)")
    numeric_cols = ['price','rating']
    trending_avg = trending[numeric_cols].mean().to_dict() if not trending.empty else {c:None for c in numeric_cols}
    non_trending_avg = non_trending[numeric_cols].mean().to_dict() if not non_trending.empty else {c:None for c in numeric_cols}

    cols_left, cols_right = st.columns(2)
    cols_left.write("Trending items averages")
    cols_left.write(trending_avg)
    cols_right.write("Non-trending items averages")
    cols_right.write(non_trending_avg)

    st.markdown("---")
    top_feats = list(feature_importance.head(5)['feature'].values) if not feature_importance.empty else []
    st.write(f"Top model features: {top_feats}")

def page_predictions(predictor, scaler):
    st.markdown('<div class="sub-header">🔮 Make Predictions</div>', unsafe_allow_html=True)
    st.write("Enter item details to predict whether it will be trending.")

    # Input fields
    color = st.selectbox("Color", options=config.COLORS)
    style = st.selectbox("Style Attribute", options=config.STYLE_ATTRIBUTES)
    brand = st.selectbox("Brand", options=config.BRANDS)
    season = st.selectbox("Season", options=config.SEASONS)
    category = st.selectbox("Category", options=config.CATEGORIES)
    price = st.number_input("Price ($)", min_value=0.0, value=100.0, step=1.0)
    rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    reviews_count = st.number_input("Review count", min_value=0, value=10, step=1)
    age = st.number_input("Customer age", min_value=0, value=25, step=1)
    total_sizes = st.number_input("Total sizes available", min_value=1, value=5, step=1)
    size = st.selectbox("Size", options=config.SIZES)
    description = st.selectbox("Description", options=config.DESCRIPTIONS)
    purchase_history = st.selectbox("Purchase history", options=config.PURCHASE_HISTORY)
    fashion_magazines = st.selectbox("Fashion magazines mentioned", options=config.FASHION_MAGAZINES)
    fashion_influencers = st.selectbox("Fashion influencers", options=config.FASHION_INFLUENCERS)
    time_period_highest_purchase = st.selectbox("Time period with highest purchase", options=config.TIME_PERIODS)
    customer_reviews = st.selectbox("Customer reviews", options=config.CUSTOMER_REVIEWS)
    social_media_comments = st.selectbox("Social media comments", options=config.SOCIAL_COMMENTS)
    feedback = st.selectbox("Feedback", options=config.FEEDBACK_OPTIONS)

    if st.button("Predict"):
        # Prepare input df with all required columns
        input_dict = {c:0 for c in predictor.feature_order}
        # Fill numeric
        for col in config.NUMERIC_FEATURES:
            input_dict[col] = price if col=='price' else rating if col=='rating' else 0
        # Fill categorical
        for col in config.CATEGORICAL_FEATURES:
            val_map = {
                'brand': brand,
                'category': category,
                'color': color,
                'style attributes': style,
                'season': season,
                'fashion influencers': fashion_influencers
            }
            input_dict[col] = val_map.get(col,"unknown")
        input_df = pd.DataFrame([input_dict])

        # Scale numeric same as training
        input_df[config.NUMERIC_FEATURES] = scaler.transform(input_df[config.NUMERIC_FEATURES])

        # Predict
        pred, prob = predictor.predict(input_df)
        pred_val = int(pred[0]) if hasattr(pred, '__len__') else int(pred)
        prob_val = float(prob[0][1]) if prob is not None else None

        if pred_val==1:
            st.success("✨ PREDICTED: TRENDING")
            st.balloons()
        else:
            st.warning("❌ PREDICTED: NOT TRENDING")
        if prob_val is not None:
            st.metric("Confidence", f"{prob_val:.2%}")
        if pred_val==1:
            st.info("Recommendation: Consider boosting marketing and inventory for this item.")
        else:
            st.info("Recommendation: Improve description, feedback sentiment, or consider promotional pricing.")

# -----------------------
if __name__ == "__main__":
    main()
