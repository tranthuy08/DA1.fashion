import os, sys, time, joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

# ensure root path includes project root if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import FashionTrendPredictor
from data_generator import generate_fashion_data
import config

# LOGGER SETUP 
import io
log_buffer = io.StringIO()

class DualLogger:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

sys.stdout = DualLogger(sys.stdout, log_buffer)

# START TIMER
start_time = time.time()
print("\n TRAINING PIPELINE START \n")

# LOAD DATA 
df = generate_fashion_data()
print(f"Dataset loaded: {df.shape}")

# PREPROCESSING 
numeric_cols = config.NUMERIC_FEATURES.copy()
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# keep text cols as-is 
text_cols = config.TEXT_COLUMNS.copy()
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("")

# FEATURES & TARGET
categorical_cols = config.CATEGORICAL_FEATURES.copy()
X = df[numeric_cols + categorical_cols]
y = df['Trend']
cat_indices = [X.columns.get_loc(c) for c in categorical_cols]

# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
)

# TRAIN USING CLASS MODEL
predictor = FashionTrendPredictor(
    iterations=config.CATBOOST_ITERATIONS,
    learning_rate=config.CATBOOST_LEARNING_RATE,
    depth=config.CATBOOST_DEPTH
)

predictor.train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_test,
    y_val=y_test,
    cat_features_idx=cat_indices
)

# EVALUATION
metrics = predictor.evaluate(X_test, y_test)

# ROC-AUC 
try:
    y_proba = predictor.model.predict_proba(X_test)[:,1]
    metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
except Exception:
    metrics['roc_auc'] = None

print("\nModel Evaluation:")
for k, v in metrics.items():
    if k not in ['confusion_matrix', 'classification_report']:
        print(f"{k}: {v}")

print("\nConfusion Matrix:\n", metrics['confusion_matrix'])
print("\nClassification Report:\n", metrics['classification_report'])

# SAVE MODEL + SCALER 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fashion_model_output", timestamp)
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "fashion_trend_model.cbm")
predictor.model.save_model(model_path)

scaler_path = os.path.join(model_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)

print(f"\nModel saved to: {model_dir}")

# PLOTS

# Confusion Matrix 
plt.figure(figsize=(6, 5))
sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
plt.close()

# Feature Importance
feature_importance_df = predictor.get_feature_importance()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance_df.head(15),
    x="importance",
    y="feature"
)
plt.title("Top 15 Feature Importance (CatBoost)")
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "feature_importance.png"))
plt.close()

# SHAP Summary
explainer = shap.TreeExplainer(predictor.model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False, max_display=15)
plt.savefig(os.path.join(model_dir, "shap_summary.png"), dpi=300, bbox_inches="tight")
plt.close()

# END TIMER
elapsed = time.time() - start_time
print(f"\nTraining pipeline finished in {elapsed:.2f} seconds")
print("\n TRAINING COMPLETE \n")

# SAVE LOG
log_path = os.path.join(model_dir, "run_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write(log_buffer.getvalue())
print(f"Log saved at: {log_path}")
