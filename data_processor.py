"""FashionTrendPredictor: train, evaluate, feature importance, predict."""

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score

import config

class FashionTrendPredictor:
    def __init__(self, iterations=None, learning_rate=None, depth=None):
        self.numerical_features = config.NUMERIC_FEATURES.copy()
        self.categorical_features = config.CATEGORICAL_FEATURES.copy()
        self.feature_order = self.numerical_features + self.categorical_features
        self.cat_features_idx = [self.feature_order.index(c) for c in self.categorical_features]

        iters = iterations if iterations is not None else config.CATBOOST_ITERATIONS
        lr = learning_rate if learning_rate is not None else config.CATBOOST_LEARNING_RATE
        d = depth if depth is not None else config.CATBOOST_DEPTH

        self.model = CatBoostClassifier(
            iterations=iters,
            learning_rate=lr,
            depth=d,
            eval_metric="F1",
            verbose=100
        )

    def prepare_data(self, df):
        """
        Ensure columns exist, fill missing values, return dataframe in feature order
        and categorical indices for CatBoost.
        """
        df = df.copy()
        # numeric
        for col in self.numerical_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            else:
                df[col] = 0
        # categorical
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("unknown")
            else:
                df[col] = "unknown"

        df = df[self.feature_order]
        return df, self.cat_features_idx

    def train(self, X_train, y_train, X_val=None, y_val=None, cat_features_idx=None):
        """
        Train CatBoost model. X_train/X_val should be pandas DataFrame in feature_order.
        cat_features_idx should be list of integer indices (as used in CatBoost).
        """
        # Use cat_features_idx if provided, otherwise use self.cat_features_idx
        cat_idx = cat_features_idx if cat_features_idx is not None else self.cat_features_idx

        # If CatBoost needs Pool for eval, wrap it
        eval_set = (X_val, y_val) if (X_val is not None and y_val is not None) else None

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            cat_features=cat_idx,
            verbose=100
        )

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        y_proba = None
        try:
            y_proba = self.model.predict_proba(X)[:,1]
        except Exception:
            pass

        results = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred),
            "classification_report": classification_report(y, y_pred)
        }
        if y_proba is not None:
            results["y_proba"] = y_proba
        return results

    def get_feature_importance(self):
        fi = self.model.get_feature_importance()
        return pd.DataFrame({
            "feature": list(self.model.feature_names_),
            "importance": fi
        }).sort_values(by="importance", ascending=False)

    def predict(self, df):
        df_prepared, cat_idx = self.prepare_data(df)
        pred = self.model.predict(df_prepared)
        prob = None
        try:
            prob = self.model.predict_proba(df_prepared)
        except Exception:
            prob = None
        return pred, prob
