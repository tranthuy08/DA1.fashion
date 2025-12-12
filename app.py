"""
Fashion Trend Prediction System - Streamlit App 
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
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

from sklearn.model_selection import train_test_split

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
def load_raw_and_generated():
    """
    Return:
      raw_df: dataframe loaded directly from CSV (original values)
      df: dataframe from generate_fashion_data() (has trend_score, Trend; numeric cols scaled)
      scaler_params: dict of {col: (min, max)} from raw_df for numeric features
    """
    # Load raw data directly
    raw_df = pd.read_csv(config.DATA_FILE, encoding="utf-8-sig")

    # compute min/max for numeric features from raw data (used to scale prediction inputs)
    numeric_cols = [c for c in config.NUMERIC_FEATURES if c in raw_df.columns]
    scaler_params = {}
    for col in numeric_cols:
        cmin = float(raw_df[col].min()) if pd.api.types.is_numeric_dtype(raw_df[col]) else 0.0
        cmax = float(raw_df[col].max()) if pd.api.types.is_numeric_dtype(raw_df[col]) else 1.0
        # avoid zero division
        if cmax == cmin:
            cmax = cmin + 1.0
        scaler_params[col] = (cmin, cmax)

    # generate df with Trend (note: this will scale numeric cols internally)
    df = generate_fashion_data()

    return raw_df, df, scaler_params

@st.cache_resource(show_spinner=False)
def train_and_get_model(df):
    """Train model on df"""
    predictor = FashionTrendPredictor(
        iterations=config.CATBOOST_ITERATIONS,
        learning_rate=config.CATBOOST_LEARNING_RATE,
        depth=config.CATBOOST_DEPTH
    )

    # prepare_data returns df in correct feature order (and cat indices)
    df_prepared, cat_idx = predictor.prepare_data(df)

    X = df_prepared
    y = df['Trend']

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    predictor.train(X_train, y_train, X_val=X_test, y_val=y_test, cat_features_idx=cat_idx)

    test_metrics = predictor.evaluate(X_test, y_test)
    # try add roc_auc
    try:
        from sklearn.metrics import roc_auc_score
        y_proba = predictor.model.predict_proba(X_test)[:, 1]
        test_metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    except Exception:
        test_metrics.setdefault('roc_auc', None)

    feature_importance = predictor.get_feature_importance()
    return predictor, test_metrics, feature_importance, X_test, y_test

# Main App

def main():
    st.markdown('<div class="main-header">👗 Fashion Trend Prediction System</div>', unsafe_allow_html=True)
    st.write("Analyze, visualize and predict trending fashion items.")

    page = st.sidebar.radio("Navigation", [
        "📊 Overview",
        "📈 Data Insights",
        "🤖 Model Performance",
        "🎨 Trend Insights",
        "🔮 Make Predictions"
    ])

    st.sidebar.markdown("---")
    st.sidebar.info("Uses CatBoost. Data: product attributes, ratings, social engagement, price.")
    st.sidebar.markdown("### 👥 Team Members")
    st.sidebar.markdown(
        """
        **Nguyễn Thị Ngọc Khuê** – 2321050112  
        **Phạm Thị Thanh** – 2321050063  
        **Trần Thị Thúy** – 2321050089  
        """
    )

    with st.spinner("Loading data..."):
        raw_df, df, scaler_params = load_raw_and_generated()

    with st.spinner("Training model (cached)..."):
        predictor, metrics, feature_importance, X_test, y_test = train_and_get_model(df)

    if page == "📊 Overview":
        page_overview(raw_df, df, metrics)
    elif page == "📈 Data Insights":
        page_eda(raw_df, df)
    elif page == "🤖 Model Performance":
        page_model_performance(metrics, feature_importance)
    elif page == "🎨 Trend Insights":
        page_trend_insights(raw_df, df, feature_importance)
    elif page == "🔮 Make Predictions":
        page_predictions(predictor, scaler_params)


# PAGES

def page_overview(raw_df, df, metrics):
    st.markdown('<div class="sub-header">📊 System Overview</div>', unsafe_allow_html=True)
    summary = get_data_summary(df)  # uses df (Trend exists) — still fine

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Items", summary['total_items'])
    c1.metric("Unique Colors", summary['unique_colors'])
    c2.metric("Trending Items", summary['trending_items'])
    c2.metric("Unique Styles", summary['unique_styles'])
    c3.metric("Trending Ratio", f"{summary['trending_ratio']*100:.2f}%")
    c3.metric("Unique Brands", summary['unique_brands'])

    # Use raw_df values for avg price & avg rating (original units)
    avg_price = f"${raw_df['price'].mean():.2f}" if 'price' in raw_df.columns else "N/A"
    avg_rating = f"{raw_df['rating'].mean():.2f}" if 'rating' in raw_df.columns else "N/A"
    c4.metric("Avg Price", avg_price)
    c4.metric("Avg Rating", avg_rating)

    st.markdown("---")
    st.markdown("### Model Snapshot")
    acc = metrics.get('accuracy', np.nan)
    f1 = metrics.get('f1_score', np.nan)
    roc = metrics.get('roc_auc', None)
    st.write(f"Accuracy: {acc:.4f}" if not pd.isna(acc) else "Accuracy: N/A")
    st.write(f"F1-score: {f1:.4f}" if not pd.isna(f1) else "F1-score: N/A")
    st.write(f"ROC-AUC: {roc:.4f}" if roc is not None else "ROC-AUC: N/A")

def page_eda(raw_df, df):
    st.markdown('<div class="sub-header">📊 Data Exploration & Trend Analysis</div>', unsafe_allow_html=True)
    sns.set(style="whitegrid")
    figsize = (config.CHART_WIDTH / 100, config.CHART_HEIGHT / 100)

    # 1. Dataset Summary
    st.subheader("📁 Dataset Overview")

    st.write("Dataset shape (raw):", raw_df.shape)
    st.write("Missing values (raw):")
    st.dataframe(raw_df.isnull().sum().sort_values(ascending=False).to_frame("missing_count"))

    # 2. Filters 
    st.sidebar.markdown("### 🔎 Filters")

    trend_options = ['All'] + df['Trend'].unique().tolist() if 'Trend' in df.columns else ['All']
    trend_filter = st.sidebar.selectbox("Filter by Trend", options=trend_options, index=0)

    categories = ['All'] + df['category'].unique().tolist() if 'category' in df.columns else ['All']
    category_filter = st.sidebar.selectbox("Filter by Category", options=categories, index=0)

    brands = ['All'] + df['brand'].unique().tolist() if 'brand' in df.columns else ['All']
    brand_filter = st.sidebar.selectbox("Filter by Brand", options=brands, index=0)

    seasons = ['All'] + df['season'].unique().tolist() if 'season' in df.columns else ['All']
    season_filter = st.sidebar.selectbox("Filter by Season", options=seasons, index=0)

    df_filtered = df.copy()

    if trend_filter != 'All' and 'Trend' in df.columns:
        df_filtered = df_filtered[df_filtered['Trend'] == trend_filter]

    if category_filter != 'All':
        df_filtered = df_filtered[df_filtered['category'] == category_filter]

    if brand_filter != 'All':
        df_filtered = df_filtered[df_filtered['brand'] == brand_filter]

    if season_filter != 'All':
        df_filtered = df_filtered[df_filtered['season'] == season_filter]

    # 3. Basic EDA 
    st.subheader("📈 Basic Data Exploration")

    # Rating distribution
    if 'rating' in raw_df.columns:
        st.markdown("### Rating Distribution")
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(raw_df['rating'].dropna(), bins=30, kde=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # Top categories & brands
    st.markdown("### Top Categories & Brands")
    c1, c2 = st.columns(2)

    if 'category' in df_filtered.columns:
        top_categories = df_filtered['category'].value_counts().head(config.TOP_N)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_categories.index, y=top_categories.values, palette="Set3", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        c1.pyplot(fig)
        plt.close(fig)

    if 'brand' in df_filtered.columns:
        top_brands = df_filtered['brand'].value_counts().head(config.TOP_N)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_brands.index, y=top_brands.values, palette="Set2", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        c2.pyplot(fig)
        plt.close(fig)

    # Top N Category/Brand by Metric
    st.write("### Top Categories & Brands with Sorting Metric")
    metric_options = ['trend_score', 'rating']
    metric = st.selectbox("Sort Top N by", options=metric_options, index=0)
    c1, c2 = st.columns(2)

    raw_filtered = raw_df.loc[df_filtered.index]

    if 'category' in raw_filtered.columns and metric in raw_filtered.columns:
        top_cat_count = raw_filtered['category'].value_counts().head(config.TOP_N)
        top_cat_metric = raw_filtered.groupby('category')[metric].mean().loc[top_cat_count.index]

        top_categories_df = pd.DataFrame({
            "Count": top_cat_count,
            f"Avg {metric}": top_cat_metric
        })
        c1.dataframe(top_categories_df)

    if 'brand' in df_filtered.columns and metric in df_filtered.columns:
        top_brands_count = df_filtered['brand'].value_counts().head(config.TOP_N)
        top_brands_metric = df_filtered.groupby('brand')[metric].mean().loc[top_brands_count.index]
        top_brands_df = pd.DataFrame({
            "Count": top_brands_count,
            f"Avg {metric}": top_brands_metric
        })
        c2.write("**Top Brands Table**")
        c2.dataframe(top_brands_df, use_container_width=True)

    # Word Clouds
    st.subheader("☁ Word Clouds")
    text_cols = [c for c in config.TEXT_COLUMNS if c in raw_df.columns]

    if text_cols:
        for col in text_cols:
            text = " ".join(raw_df[col].dropna().astype(str).values)
            if text.strip():
                wc = WordCloud(
                    width=int(config.CHART_WIDTH),
                    height=int(config.CHART_HEIGHT/2),
                    background_color="white"
                ).generate(text)

                fig, ax = plt.subplots(figsize=(config.CHART_WIDTH/100, config.CHART_HEIGHT/100))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title(f"Word Cloud: {col}")
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("No textual columns available for wordclouds.")

    # 4. Trend Analysis
    st.markdown("---")
    st.subheader("📊 Trend Analysis")

    # Trend distribution
    try:
        st.plotly_chart(viz.plot_trend_distribution(df_filtered), use_container_width=True)
    except:
        pass

    col1, col2 = st.columns(2)

    # Color Trend
    with col1:
        try:
            st.plotly_chart(viz.plot_color_trends(df_filtered), use_container_width=True)
        except:
            pass

    # Style Trends & Seasonal Trends
    with col2:
        try:
            st.plotly_chart(viz.plot_style_trends(df_filtered), use_container_width=True)
        except:
            pass
        try:
            st.plotly_chart(viz.plot_seasonal_trends(df_filtered), use_container_width=True)
        except:
            pass

    # 5. Sample Rows
    st.markdown("---")
    st.subheader("📋 Sample Rows")

    raw_filtered = raw_df.loc[df_filtered.index]
    st.dataframe(raw_filtered.head(30), use_container_width=True)


def page_model_performance(metrics, feature_importance):
    st.markdown('<div class="sub-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    try:
        st.plotly_chart(viz.plot_metrics_summary(metrics), use_container_width=True)
    except Exception:
        st.write(metrics)

    left, right = st.columns(2)
    with left:
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
        try:
            st.plotly_chart(viz.plot_confusion_matrix(metrics['confusion_matrix']), use_container_width=True)
        except Exception:
            st.write(metrics.get('confusion_matrix', 'N/A'))
    with right:
        st.text(metrics.get('classification_report', 'N/A'))
        try:
            st.plotly_chart(viz.plot_feature_importance(feature_importance), use_container_width=True)
            st.dataframe(feature_importance.head(10), use_container_width=True)
        except Exception:
            st.write(feature_importance.head(10))

def page_trend_insights(raw_df, df, feature_importance):
    st.markdown('<div class="sub-header">🎨 Trend Insights</div>', unsafe_allow_html=True)
    # Use df to get Trend index, but compute averages on raw_df to get original units
    if 'Trend' not in df.columns:
        st.info("Trend labels missing.")
        return

    trending_idx = df[df['Trend'] == 1].index
    non_idx = df[df['Trend'] == 0].index

    trending_raw = raw_df.loc[trending_idx] if not raw_df.empty else pd.DataFrame()
    non_raw = raw_df.loc[non_idx] if not raw_df.empty else pd.DataFrame()

    st.markdown("Top colors/styles/brands among trending (raw counts):")
    c1, c2, c3 = st.columns(3)
    if 'color' in raw_df.columns:
        c1.write(raw_df.loc[trending_idx, 'color'].value_counts().head(10))
    if 'style attributes' in raw_df.columns:
        c2.write(raw_df.loc[trending_idx, 'style attributes'].value_counts().head(10))
    if 'brand' in raw_df.columns:
        c3.write(raw_df.loc[trending_idx, 'brand'].value_counts().head(10))

    st.markdown("---")
    st.markdown("Statistical Summary (use raw values):")
    cols_left, cols_right = st.columns(2)
    with cols_left:
        st.write("Trending items averages")
        if not trending_raw.empty:
            st.write({
                "avg_price": float(trending_raw['price'].mean()) if 'price' in trending_raw.columns else None,
                "avg_rating": float(trending_raw['rating'].mean()) if 'rating' in trending_raw.columns else None
            })
        else:
            st.write("No trending items (raw).")
    with cols_right:
        st.write("Non-trending items averages")
        if not non_raw.empty:
            st.write({
                "avg_price": float(non_raw['price'].mean()) if 'price' in non_raw.columns else None,
                "avg_rating": float(non_raw['rating'].mean()) if 'rating' in non_raw.columns else None
            })
        else:
            st.write("No non-trending items (raw).")

    st.markdown("---")
    st.write("Top model features (from feature importance):")
    st.write(list(feature_importance.head(10)['feature'].values)) if not feature_importance.empty else st.write([])

def page_predictions(predictor, scaler_params):
    st.markdown('<div class="sub-header">🔮 Make Predictions</div>', unsafe_allow_html=True)

    # Use options from config where possible
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

    # Fields previously text now selectboxes
    description = st.selectbox("Description", options=config.DESCRIPTIONS)
    purchase_history = st.selectbox("Purchase history", options=config.PURCHASE_HISTORY)
    fashion_magazines = st.selectbox("Fashion magazines mentioned", options=config.FASHION_MAGAZINES)
    fashion_influencers = st.selectbox("Fashion influencers", options=config.FASHION_INFLUENCERS)
    time_period_highest_purchase = st.selectbox("Time period with highest purchase", options=config.TIME_PERIODS)
    customer_reviews = st.selectbox("Customer reviews", options=config.CUSTOMER_REVIEWS)
    social_media_comments = st.selectbox("Social media comments", options=config.SOCIAL_COMMENTS)
    feedback = st.selectbox("Feedback", options=config.FEEDBACK_OPTIONS)

    if st.button("Predict"):
        # Prepare input dataframe
        input_df = pd.DataFrame([{
            'price': price,
            'brand': brand,
            'category': category,
            'description': description,
            'rating': rating,
            'review count': reviews_count,
            'style attributes': style,
            'total sizes': total_sizes,
            'size': size,
            'color': color,
            'purchase history': purchase_history,
            'age': age,
            'fashion magazines': fashion_magazines,
            'fashion influencers': fashion_influencers,
            'season': season,
            'time period highest purchase': time_period_highest_purchase,
            'customer reviews': customer_reviews,
            'social media comments': social_media_comments,
            'feedback': feedback,
        }])

        # Scale numeric columns using scaler_params computed from raw_df
        for col, (vmin, vmax) in scaler_params.items():
            if col in input_df.columns:
                val = float(input_df.loc[0, col])
                val_clipped = max(min(val, vmax), vmin)
                scaled = (val_clipped - vmin) / (vmax - vmin) if (vmax - vmin) != 0 else 0.0
                input_df.loc[0, col] = scaled

        # Predict (predictor.predict will call prepare_data internally)
        pred, prob = predictor.predict(input_df)

        pred_val = int(np.asarray(pred)[0]) if hasattr(pred, '__len__') else int(pred)

        prob_val = None
        if prob is not None:
            try:
                prob_val = float(prob[0][1])
            except:
                try:
                    prob_val = float(prob[0])
                except:
                    prob_val = None

        if pred_val == 1:
            st.success("✨ PREDICTED: TRENDING")
            st.balloons()
        else:
            st.warning("❌ PREDICTED: NOT TRENDING")

        if prob_val is not None:
            st.metric("Confidence", f"{prob_val:.2%}")

        if pred_val == 1:
            st.info("Recommendation: consider increasing inventory and marketing exposure.")
        else:
            st.info("Recommendation: consider improving engagement, pricing or bundling options.")

if __name__ == "__main__":
    main()

