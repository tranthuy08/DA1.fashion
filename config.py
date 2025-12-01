"""Configuration parameters for the fashion trend prediction system."""

# DATA PARAMETERS
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

DATA_FILE = "cleaned_mock_fashion_data.csv"

# DATASET CATEGORIES
BRANDS = [
    'ted baker', 'alexander mcqueen', 'tommy hilfiger', 'ralph lauren',
    'calvin klein', 'jigsaw', 'mulberry', 'burberry'
]

CATEGORIES = [
    'tops', 'outerwear', 'bottoms', 'dresses',
    'swimwear', 'activewear', 'lingerie',
]

COLORS = [
    'black', 'red', 'green', 'blue'
]

STYLE_ATTRIBUTES = [
    'vintage', 'formal', 'sporty', 'streetwear', 'edgy', 'minimalist',
    'preppy', 'glamorous', 'casual', 'bohemian'
]

SEASONS = ['winter', 'fall/winter', 'spring', 'summer', 'spring/summer', 'fall']

FASHION_INFLUENCERS = [
    'leandra medine', 'chiara ferragni', 'song of style',
    'olivia palermo', 'kendall jenner', 'aimee song',
    'negin mirsalehi', 'gigi hadid', 'julie sariñana', 'camila coelho'
]

# New fields for Make Predictions selectboxes
DESCRIPTIONS = [
    'bad', 'not good', 'very bad', 'very good', 'best', 'good','worst'
]
SIZES = ['S', 'M', 'L', 'XL']
PURCHASE_HISTORY = [
    'medium', 'above average', 'average', 'very high', 'negligible',
       'very low', 'significant', 'below average', 'low', 'high'
]
FASHION_MAGAZINES = ['vogue', 'glamour', 'marie claire', 'fashionista', 'w',
       "harper's bazaar", 'grazia', 'cosmopolitan', 'elle', 'instyle']
TIME_PERIODS = ['daytime', 'weekend', 'nighttime', 'holiday', 'evening']
CUSTOMER_REVIEWS = ['mixed', 'negative', 'unknown', 'neutral', 'positive']
SOCIAL_COMMENTS = ['mixed', 'neutral', 'negative', 'other', 'positive', 'unknown']
FEEDBACK_OPTIONS = ['other', 'neutral', 'positive', 'negative', 'unknown', 'mixed']

# These fields exist and are numeric in the cleaned dataset
NUMERIC_FEATURES = ['rating', 'review count', 'price', 'age']

CATEGORICAL_FEATURES = [
    'brand', 'category', 'color', 'style attributes',
    'season', 'fashion influencers'
]

TEXT_COLUMNS = [
    'feedback',
    'social media comments',
    'purchase history'
]

# MODEL PARAMETERS
CATBOOST_ITERATIONS = 500
CATBOOST_LEARNING_RATE = 0.1
CATBOOST_DEPTH = 6
CATBOOST_LOSS = "Logloss"
CATBOOST_EVAL = "AUC"
EARLY_STOPPING = 50

# TREND LABELING CONFIG 
TREND_SCORE_WEIGHTS = {
    "rating": 0.4,
    "review count": 0.35,
    "price": 0.15,
    "age": 0.1
}

TREND_THRESHOLD_QUANTILE = 0.70

# VISUALIZATION
CHART_WIDTH = 800
CHART_HEIGHT = 400
COLOR_PALETTE = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#FFA07A', '#556270']

# EDA options
TOP_N = 10
