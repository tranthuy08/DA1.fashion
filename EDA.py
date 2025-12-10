import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import config
from data_generator import generate_fashion_data

# CONFIG
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
output_dir = "eda_outputs"
os.makedirs(output_dir, exist_ok=True)

# LOAD DATA
raw_df = pd.read_csv(config.DATA_PATH)
df = generate_fashion_data()
print("Data loaded successfully.")
print(df.info(), "\n")

# BASIC OVERVIEW 
print("Missing values per column:")
print(df.isnull().sum().sort_values(ascending=False))
print("\nDescriptive statistics:")
print(df.describe(include='all'))

# Save basic info to text file
eda_text_path = os.path.join(output_dir, "eda_basic_info.txt")
with open(eda_text_path, "w", encoding="utf-8") as f:
    f.write("=== BASIC DATA INFORMATION ===\n\n")
    f.write(f"Dataset shape: {df.shape}\n\n")
    f.write(">>> Data Info:\n")
    df.info(buf=f)
    f.write("\n\n>>> Missing Values:\n")
    f.write(str(df.isnull().sum().sort_values(ascending=False)) + "\n\n")
    f.write(">>> Descriptive Statistics:\n")
    f.write(str(df.describe(include='all')) + "\n")

print(f"Basic dataset info saved to: {eda_text_path}")

# VISUALIZATIONS 
def get_figsize():
    return (config.CHART_WIDTH/100, config.CHART_HEIGHT/100)

top_n = config.TOP_N
colors = config.COLOR_PALETTE

# 1. Distribution of Rating
plt.figure(figsize=get_figsize())
sns.histplot(raw_df['rating'], bins=30, kde=True, color=colors[0])
plt.title("Distribution of Product Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rating_distribution.png"), dpi=300)
plt.close()


# MATCH RAW + DF
df_plot = pd.DataFrame({
    "Trend": df["Trend"],
    "rating": raw_df["rating"]
})

# 2. Boxplot Rating by Trend
plt.figure(figsize=get_figsize())
sns.boxplot(x='Trend', y='rating', data=df_plot, palette=colors[:2])
plt.title("Rating Distribution by Product Trend")
plt.xlabel("Product Trend (0 = Not Trending, 1 = Trending)")
plt.ylabel("Rating")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rating_by_trend.png"), dpi=300)
plt.close()


# 3. Top N Categories
plt.figure(figsize=get_figsize())
top_categories = df['category'].value_counts().head(top_n)
sns.barplot(x=top_categories.values, y=top_categories.index, palette=colors)
plt.title(f"Top {top_n} Product Categories")
plt.xlabel("Count")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_categories.png"), dpi=300)
plt.close()

# 4. Top N Brands
plt.figure(figsize=get_figsize())
top_brands = df['brand'].value_counts().head(top_n)
sns.barplot(x=top_brands.values, y=top_brands.index, palette=colors)
plt.title(f"Top {top_n} Brands")
plt.xlabel("Count")
plt.ylabel("Brand")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_brands.png"), dpi=300)
plt.close()

# 5. Trend Distribution by Season
plt.figure(figsize=get_figsize())
sns.countplot(x='season', hue='Trend', data=df, palette=colors[:2])
plt.title("Product Trend Distribution by Season")
plt.xlabel("Season")
plt.ylabel("Count")
plt.legend(title="Product Trend")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "trend_by_season.png"), dpi=300)
plt.close()

# 6. Correlation Matrix
plt.figure(figsize=get_figsize())
sns.heatmap(df[['trend_score', 'Trend']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix: Trend Score vs Product Trend")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
plt.close()

# 7. WordClouds
text_sources = {
    "style_attributes": "style attributes",
    "fashion_influencers": "fashion influencers",
    "social_media_comments": "social media comments"
}
for key, col in text_sources.items():
    if col in df.columns:
        text = " ".join(df[col].dropna().astype(str))
        if text.strip():
            wc = WordCloud(width=config.CHART_WIDTH, height=int(config.CHART_HEIGHT/2),
                           background_color="white").generate(text)
            plt.figure(figsize=get_figsize())
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Word Cloud: {col.title()}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"wordcloud_{key}.png"), dpi=300)
            plt.close()

# 8. Label Balance
plt.figure(figsize=get_figsize())
sns.countplot(x='Trend', data=df, palette=colors[:2])
plt.title("Distribution of Product Trend Labels")
plt.xlabel("Product Trend (0 = Not Trending, 1 = Trending)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "label_balance.png"), dpi=300)
plt.close()

# SUMMARY LOG 
summary_path = os.path.join(output_dir, "eda_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("EDA Summary Report\n")
    f.write("==================\n\n")
    f.write(f"Dataset shape: {df.shape}\n\n")

    f.write("Top categories:\n")
    f.write(str(top_categories) + "\n\n")

    f.write("Top brands:\n")
    f.write(str(top_brands) + "\n\n")

    f.write("Rating statistics (REAL):\n")
    f.write(str(raw_df["rating"].describe()) + "\n\n")

    f.write("Label distribution:\n")
    f.write(str(df['Trend'].value_counts()) + "\n")


print(f"\nAll EDA visualizations and summary saved in: {output_dir}/")
