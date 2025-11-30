"""
Fashion Trend Prediction System - Streamlit App
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Streamlit page config
st.set_page_config(page_title="Fashion Trend Prediction System", page_icon="👗", layout="wide")

# Simple styling
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #FF6B6B; text-align: center; padding: 0.5rem; }
    .sub-header { font-size: 1.25rem; color: #4ECDC4; margin-top: 0.5rem; }
    .metric-card { background-color: #f7fbfc; padding: 0.8rem; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    """Load data via shared generator so Trend label is consistent."""
    df = generate_fashion_data()
    return df


@st.cache_resource(show_spinner=False)
def train_and_get_model(df):
    """
    Train CatBoost model (using FashionTrendPredictor class).
    Returns trained predictor, test metrics, feature importance and test split.
    """
    predictor = FashionTrendPredictor(
        iterations=config.CATBOOST_ITERATIONS,
        learning_rate=config.CATBOOST_LEARNING_RATE,
        depth=config.CATBOOST_DEPTH
    )

    # Prepare data using predictor.prepare_data (it ensures required cols)
    df_prepared, cat_idx = predictor.prepare_data(df)

    # X and y
    X = df_prepared
    y = df['Trend']

    # Train/Test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )

    # Fit model (CatBoost)
    predictor.train(X_train, y_train, X_val=X_test, y_val=y_test, cat_features_idx=cat_idx)

    # Metrics and feature importance
    test_metrics = predictor.evaluate(X_test, y_test)
    # try to add roc_auc if available (some evaluate() may include y_proba)
    try:
        from sklearn.metrics import roc_auc_score
        y_proba = predictor.model.predict_proba(X_test)[:, 1]
        test_metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    except Exception:
        test_metrics.setdefault('roc_auc', None)

    feature_importance = predictor.get_feature_importance()

    return predictor, test_metrics, feature_importance, X_test, y_test


def main():
    # Header
    st.markdown('<div class="main-header">👗 Fashion Trend Prediction System</div>', unsafe_allow_html=True)
    st.write("Powered by CatBoost — analyze, visualize and predict trending fashion items.")

    # Sidebar navigation
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

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()

    # Train model resource (cached)
    with st.spinner("Training model (cached)..."):
        predictor, metrics, feature_importance, X_test, y_test = train_and_get_model(df)

    # Route pages
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
        page_predictions(predictor, df)


# Page: Overview
def page_overview(df, metrics):
    st.markdown('<div class="sub-header">📊 System Overview</div>', unsafe_allow_html=True)

    summary = get_data_summary(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Items", summary['total_items'])
    c1.metric("Unique Colors", summary['unique_colors'])

    c2.metric("Trending Items", summary['trending_items'])
    c2.metric("Unique Styles", summary['unique_styles'])

    c3.metric("Trending Ratio", summary['trending_ratio'])
    c3.metric("Unique Brands", summary['unique_brands'])

    c4.metric("Avg Price", summary['avg_price'])
    c4.metric("Avg Rating", summary['avg_rating'])

    st.markdown("---")
    # Model snapshot
    st.markdown("### Model Snapshot")
    cols = st.columns(2)
    acc = metrics.get('accuracy', 0)
    f1 = metrics.get('f1_score', 0)
    roc = metrics.get('roc_auc', None)
    cols[0].write(f"**Accuracy:** {acc:.4f}" if acc is not None else "Accuracy: N/A")
    cols[0].write(f"**F1-score:** {f1:.4f}" if f1 is not None else "F1: N/A")
    cols[1].write(f"**ROC-AUC:** {roc:.4f}" if roc is not None else "ROC-AUC: N/A")

    st.markdown("---")
    st.markdown("### Quick Insights")
    trending = df[df['Trend'] == 1]
    if not trending.empty:
        top_color = trending['color'].mode().iloc[0] if 'color' in trending.columns else "N/A"
        top_style = trending['style attributes'].mode().iloc[0] if 'style attributes' in trending.columns else "N/A"
        top_brand = trending['brand'].mode().iloc[0] if 'brand' in trending.columns else "N/A"
        c1, c2, c3 = st.columns(3)
        c1.info(f"🎨 Top Trending Color\n\n{top_color}")
        c2.info(f"👔 Top Trending Style\n\n{top_style}")
        c3.info(f"🏷️ Top Trending Brand\n\n{top_brand}")
    else:
        st.info("No trending items found in dataset.")


# Page: EDA Report 
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


# Page: Data Analysis 
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

    st.markdown("---")
    st.markdown("### Sample Rows")
    st.dataframe(df.sample(min(200, len(df))).reset_index(drop=True), use_container_width=True)


# Page: Model Performance
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


# Page: Trend Insights
def page_trend_insights(df, feature_importance):
    st.markdown('<div class="sub-header">🎨 Trend Insights</div>', unsafe_allow_html=True)
    trending = df[df["Trend"] == 1]
    non = df[df["Trend"] == 0]

    if trending.empty:
        st.info("No trending items detected.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Top Colors")
        st.write(trending["color"].value_counts().head(10))
    with c2:
        st.markdown("#### Top Styles")
        st.write(trending["style attributes"].value_counts().head(10))
    with c3:
        st.markdown("#### Top Brands")
        st.write(trending["brand"].value_counts().head(10))

    st.markdown("---")
    st.markdown("#### Statistical Summary (Trending vs Non-Trending)")

    col_left, col_right = st.columns(2)

    with col_left:
        st.write("### ⭐ Trending items averages")
        st.metric("Avg Price", f"${trending['price'].mean():.2f}")
        st.metric("Avg Rating", f"{trending['rating'].mean():.2f}")

    with col_right:
        st.write("### ⚪ Non-trending items averages")
        st.metric("Avg Price", f"${non['price'].mean():.2f}")
        st.metric("Avg Rating", f"{non['rating'].mean():.2f}")

    st.markdown("---")
    st.markdown("### Key Features Driving Trend")
    top_feats = list(feature_importance.head(5)["feature"].values)
    st.write(f"Top model features: {top_feats}")


# Page: Make Predictions
def page_predictions(predictor, df):
    st.markdown('<div class="sub-header">🔮 Make Predictions</div>', unsafe_allow_html=True)
    st.write("Enter item details to predict whether it will be trending.")

    # Use options from config where possible
    color = st.selectbox("Color", options=config.COLORS if hasattr(config, 'COLORS') else [])
    style = st.selectbox("Style Attribute", options=config.STYLE_ATTRIBUTES if hasattr(config, 'STYLE_ATTRIBUTES') else [])
    brand = st.selectbox("Brand", options=config.BRANDS if hasattr(config, 'BRANDS') else [])
    season = st.selectbox("Season", options=config.SEASONS if hasattr(config, 'SEASONS') else [])
    category = st.selectbox("Category", options=config.CATEGORIES if hasattr(config, 'CATEGORIES') else [])

    price = st.number_input("Price ($)", min_value=0.0, value=100.0, step=1.0)
    rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    reviews_count = st.number_input("Review count", min_value=0, value=10, step=1)

    description = st.text_area("Description", value="")
    total_sizes = st.number_input("Total sizes available", min_value=1, value=5, step=1)
    size = st.text_input("Size (e.g., M, L, XL)", value="M")

    purchase_history = st.text_area("Purchase history", value="")
    age = st.number_input("Customer age", min_value=0, value=25, step=1)

    fashion_magazines = st.text_input("Fashion magazines mentioned", value="")
    fashion_influencers = st.text_input("Fashion influencers (comma-separated)", value="")

    time_period_highest_purchase = st.text_input("Time period with highest purchase", value="")

    customer_reviews = st.text_area("Customer reviews (summarized text)", value="")
    social_media_comments = st.text_area("Social media comments (summary)", value="")
    feedback = st.text_area("Feedback", value="")

    if st.button("Predict"):
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

        # predictor.predict will internally call prepare_data
        pred, prob = predictor.predict(input_df)
        pred_val = int(pred[0]) if hasattr(pred, '__len__') else int(pred)

        # Probability extraction
        prob_val = None
        if prob is not None:
            try:
                prob_val = float(prob[0][1]) if hasattr(prob[0], '__len__') else float(prob[0])
            except:
                try:
                    prob_val = float(prob[0])
                except:
                    prob_val = None

        # Prediction result
        if pred_val == 1:
            st.success("✨ PREDICTED: TRENDING")
            st.balloons()
        else:
            st.warning("❌ PREDICTED: NOT TRENDING")

        if prob_val is not None:
            st.metric("Confidence", f"{prob_val:.2%}")

        # Recommendations
        if pred_val == 1:
            st.info("Recommendation: Consider boosting marketing and inventory for this item.")
        else:
            st.info("Recommendation: Improve description, feedback sentiment, or consider promotional pricing.")

if __name__ == "__main__":
    main()
