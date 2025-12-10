"""Visualization functions for Fashion Trend Prediction System."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def _ensure_trend(df):
    if 'Trend' not in df.columns:
        raise ValueError("Dataframe must contain 'Trend' column (use generate_fashion_data()).")

def _expand_colors(n):
    """Repeat color palette to match required length."""
    base = config.COLOR_PALETTE
    return (base * (n // len(base) + 1))[:n]

# 1. Trend Distribution
def plot_trend_distribution(df):
    _ensure_trend(df)
    counts = df['Trend'].value_counts()
    fig = go.Figure(data=[go.Bar(
        x=['Non-Trending', 'Trending'],
        y=[counts.get(0,0), counts.get(1,0)],
        marker_color=[config.COLOR_PALETTE[2], config.COLOR_PALETTE[0]],
        text=[counts.get(0,0), counts.get(1,0)],
        textposition='auto'
    )])
    fig.update_layout(title="Distribution of Trending vs Non-Trending Items",
                      xaxis_title="Category", yaxis_title="Count",
                      height=config.CHART_HEIGHT, showlegend=False)
    return fig

# 2. Top Trending Colors
def plot_color_trends(df, top_n=10):
    _ensure_trend(df)
    top_colors = df[df['Trend']==1]['color'].value_counts().head(top_n)

    colors = _expand_colors(len(top_colors))

    fig = go.Figure(data=[go.Bar(
        x=top_colors.index, 
        y=top_colors.values,
        marker_color=colors,
        text=top_colors.values, 
        textposition='auto'
    )])

    fig.update_layout(
        title="Top Trending Colors", 
        xaxis_title="Color", 
        yaxis_title="Count", 
        height=config.CHART_HEIGHT
    )
    return fig

# 3. Top Trending Styles
def plot_style_trends(df, top_n=10):
    _ensure_trend(df)
    top_styles = df[df['Trend']==1]['style attributes'].value_counts().head(top_n)

    colors = _expand_colors(len(top_styles))

    fig = go.Figure(data=[go.Bar(
        x=top_styles.index, 
        y=top_styles.values, 
        marker_color=colors,
        text=top_styles.values, 
        textposition='auto'
    )])
    fig.update_layout(
        title="Top Trending Styles", 
        xaxis_title="Style Attribute", 
        yaxis_title="Count", 
        height=config.CHART_HEIGHT
    )
    return fig

# 4. Trending Brands
def plot_brand_trends(df, top_n=10):
    _ensure_trend(df)
    top_brands = df[df['Trend']==1]['brand'].value_counts().head(top_n)

    colors = _expand_colors(len(top_brands))

    fig = go.Figure(data=[go.Bar(
        x=top_brands.index, 
        y=top_brands.values, 
        marker_color=colors,
        text=top_brands.values, 
        textposition='auto'
    )])
    fig.update_layout(
        title="Top Trending Brands", 
        xaxis_title="Brand", 
        yaxis_title="Count", 
        height=config.CHART_HEIGHT
    )
    return fig

# 5. Feature Importance
def plot_feature_importance(fi_df):
    top_features = fi_df.head(10)
    fig = go.Figure(data=[go.Bar(
        x=top_features['importance'], y=top_features['feature'],
        orientation='h', marker_color='#45B7D1',
        text=top_features['importance'].round(2), textposition='auto'
    )])
    fig.update_layout(title="Top Feature Importance", xaxis_title="Importance", yaxis_title="Feature", height=500, yaxis={'categoryorder':'total ascending'})
    return fig

# 6. Confusion Matrix
def plot_confusion_matrix(cm):
    labels = ['Non-Trending','Trending']
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, colorscale='Blues', text=cm, texttemplate='%{text}', textfont={"size":16}, showscale=True
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", height=400, width=500)
    return fig

# 7. ROC-AUC Placeholder
def plot_roc_curve_placeholder(metrics):
    fig = go.Figure()
    fig.add_annotation(text=f"ROC-AUC Score: {metrics.get('roc_auc', float('nan')):.4f}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20))
    fig.update_layout(title="Model Performance - ROC-AUC", height=300, showlegend=False)
    return fig

# 8. Seasonal Trends
def plot_seasonal_trends(df):
    _ensure_trend(df)
    season_counts = df.groupby(['season','Trend']).size().unstack(fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Non-Trending', x=season_counts.index, y=season_counts[0], marker_color=config.COLOR_PALETTE[2]))
    fig.add_trace(go.Bar(name='Trending', x=season_counts.index, y=season_counts[1], marker_color=config.COLOR_PALETTE[0]))
    fig.update_layout(title="Trending Items by Season", xaxis_title="Season", yaxis_title="Count", barmode='group', height=config.CHART_HEIGHT)
    return fig

# 9. Metrics Summary
def plot_metrics_summary(metrics):
    metric_names = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']
    metric_values = [
        metrics.get('accuracy', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1_score', 0),
        metrics.get('roc_auc', 0)
    ]
    fig = go.Figure(data=[go.Bar(x=metric_names, y=metric_values, marker_color=config.COLOR_PALETTE[:5], text=[f"{v:.3f}" for v in metric_values], textposition='auto')])
    fig.update_layout(title="Model Performance Metrics", xaxis_title="Metric", yaxis_title="Score", yaxis_range=[0,1], height=config.CHART_HEIGHT)
    return fig
