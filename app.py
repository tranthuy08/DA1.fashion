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


# Load & Train functions
@st.cache_data(show_spinner=False)
def load_data():
    df = generate_fashion_data()
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

    # Save scaler
    scaler = MinMaxScaler()
    scaler.fit(df[config.NUMERIC_FEATURES])

    return predictor, test_metrics, feature_importance, scaler, X_test, y_test

# Main App
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
    st.sidebar.info("""
        This dashboard predicts whether an item is *trending* using a CatBoost model.
        Data sources include product attributes, ratings, social engagement and prices.
    """)

    with st.spinner("Loading data..."):
        df = load_data()

    with st.spinner("Training model (cached)..."):
        predictor, metrics, feature_importance, scaler, X_test, y_test = train_and_get_model(df)

    if page == "📊 Overview":
        page_overview(df, metrics)
    elif page == "📈 EDA Report":
        page_eda(df)
    elif page == "🔍 Data Analysis":
        page_data_analysis(df)
    elif page == "🤖 Model Performance":
        page_model_performance(metrics, feature_importance)
    elif page == "🎨 Trend Insights":
        page_trend_insights(df, feature_importance)
    elif page == "🔮 Make Predictions":
        page_predictions(predictor, scaler)

# PAGES
def page_overview(df, metrics):
    st.markdown('<div class="sub-header">📊 System Overview</div>', unsafe_allow_html=True)

    trending = df[df['Trend']==1]
    non_trending = df[df['Trend']==0]

    summary = get_data_summary(df)

    numeric_cols = ['price','rating']
    trending_avg = {col: trending[f"{col}_orig"].mean() for col in numeric_cols}
    non_trending_avg = {col: non_trending[f"{col}_orig"].mean() for col in numeric_cols}

    avg_price = trending_avg['price']
    avg_rating = trending_avg['rating']

    total_items = summary.get('total_items', 0)
    unique_colors = summary.get('unique_colors', 0)
    trending_items = summary.get('trending_items', 0)
    unique_styles = summary.get('unique_styles', 0)
    trending_ratio = summary.get('trending_ratio', 0)
    unique_brands = summary.get('unique_brands', 0)

    trending_ratio_str = f"{float(trending_ratio)*100:.2f}%" if trending_ratio is not None else "N/A"
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

def page_eda(df):
    st.markdown('<div class="sub-header">📈 EDA Report</div>', unsafe_allow_html=True)
    sns.set(style="whitegrid")
    figsize = (config.CHART_WIDTH / 100, config.CHART_HEIGHT / 100)

    st.markdown("### 1) Basic Info & Missing Values")
    st.write(f"Dataset shape: {df.shape}")
    st.write("Missing values:")
    st.dataframe(df.isnull().sum().sort_values(ascending=False).to_frame("missing_count"))

    # Distribution of rating 
    if 'rating' in df.columns:
        st.markdown("### 2) Distribution of Ratings")
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(df['rating'].dropna(), bins=30, kde=True, ax=ax)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    # Rating by Trend
    if 'rating' in df.columns and 'Trend' in df.columns:
        st.markdown("### 3) Rating by Trend")
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x='Trend', y='rating', data=df, ax=ax)
        ax.set_xlabel("Trend (0 = Not Trending, 1 = Trending)")
        st.pyplot(fig)
        plt.close(fig)

    # Top categories & brands
    st.markdown("### 4) Top Categories & Brands")
    cols = st.columns(2)
    if 'category' in df.columns:
        top_cats = df['category'].value_counts().head(config.TOP_N)
        cols[0].bar_chart(top_cats)
    if 'brand' in df.columns:
        top_brands = df['brand'].value_counts().head(config.TOP_N)
        cols[1].bar_chart(top_brands)

    # Trend distribution by season
    if 'season' in df.columns and 'Trend' in df.columns:
        st.markdown("### 5) Trend Distribution by Season")
        fig, ax = plt.subplots(figsize=figsize)
        sns.countplot(x='season', hue='Trend', data=df, ax=ax)
        ax.set_xlabel("Season")
        st.pyplot(fig)
        plt.close(fig)

    # Correlation between trend_score and Trend if exists
    if {'trend_score', 'Trend'}.issubset(df.columns):
        st.markdown("### 6) Correlation: trend_score vs Trend")
        corr = df[['trend_score', 'Trend']].corr().iloc[0, 1]
        st.write(f"Pearson correlation = {corr:.3f}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[['trend_score', 'Trend']].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # Wordclouds for textual columns
    st.markdown("### 7) Word Clouds (text fields)")
    text_cols = [c for c in config.TEXT_COLUMNS if c in df.columns]
    if text_cols:
        for col in text_cols:
            text = " ".join(df[col].dropna().astype(str).values)
            if text.strip():
                wc = WordCloud(width=int(config.CHART_WIDTH), height=int(config.CHART_HEIGHT/2), background_color="white").generate(text)
                fig, ax = plt.subplots(figsize=(config.CHART_WIDTH/100, config.CHART_HEIGHT/200))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title(f"Word Cloud: {col}")
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("No textual columns available for wordclouds.")

    st.markdown("---")
    st.markdown("### Sample Data")
    st.dataframe(df.head(30), use_container_width=True)

    # download
    csv = df.to_csv(index=False)
    st.download_button("Download dataset (CSV)", csv, file_name=f"fashion_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

def page_data_analysis(df):
    st.markdown('<div class="sub-header">🔍 Data Analysis</div>', unsafe_allow_html=True)

    st.markdown("### Trend Distribution")
    try:
        fig = viz.plot_trend_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Plot error: {e}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Color Trends")
        try:
            st.plotly_chart(viz.plot_color_trends(df), use_container_width=True)
        except Exception as e:
            st.error(e)
        st.markdown("### Brand Trends")
        try:
            st.plotly_chart(viz.plot_brand_trends(df), use_container_width=True)
        except Exception as e:
            st.error(e)
    with c2:
        st.markdown("### Style Trends")
        try:
            st.plotly_chart(viz.plot_style_trends(df), use_container_width=True)
        except Exception as e:
            st.error(e)
        st.markdown("### Seasonal Trends")
        try:
            st.plotly_chart(viz.plot_seasonal_trends(df), use_container_width=True)
        except Exception as e:
            st.error(e)

def page_model_performance(metrics, feature_importance):
    st.markdown('<div class="sub-header">🤖 Model Performance</div>', unsafe_allow_html=True)

    st.markdown("### Metrics Summary")
    try:
        st.plotly_chart(viz.plot_metrics_summary(metrics), use_container_width=True)
    except Exception:
        st.write(metrics)

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("Detailed Metrics")
        dfm = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Score": [
                metrics.get('accuracy', np.nan),
                metrics.get('precision', np.nan),
                metrics.get('recall', np.nan),
                metrics.get('f1_score', np.nan),
                metrics.get('roc_auc', np.nan)
            ]
        })
        st.dataframe(dfm, use_container_width=True)

        st.markdown("Confusion Matrix")
        try:
            st.plotly_chart(viz.plot_confusion_matrix(metrics['confusion_matrix']), use_container_width=True)
        except Exception as e:
            st.write(metrics.get('confusion_matrix', 'N/A'))
    with right:
        st.markdown("Classification Report")
        st.text(metrics.get('classification_report', 'N/A'))
        st.markdown("Feature importance (top 15)")
        try:
            st.plotly_chart(viz.plot_feature_importance(feature_importance), use_container_width=True)
            st.dataframe(feature_importance.head(15), use_container_width=True)
        except Exception as e:
            st.write(feature_importance.head(15))

def page_trend_insights(df, feature_importance):
    st.markdown('<div class="sub-header">🎨 Trend Insights</div>', unsafe_allow_html=True)

    trending = df[df['Trend']==1]
    non_trending = df[df['Trend']==0]

    numeric_cols = ['price', 'rating']
    trending_avg = {col: trending[f"{col}_orig"].mean() for col in numeric_cols}
    non_trending_avg = {col: non_trending[f"{col}_orig"].mean() for col in numeric_cols}

    cols_left, cols_right = st.columns(2)
    cols_left.write("Trending items averages")
    cols_left.write(trending_avg)
    cols_right.write("Non-trending items averages")
    cols_right.write(non_trending_avg)


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


if __name__ == "__main__":
    main()
