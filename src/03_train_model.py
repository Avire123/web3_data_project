import os
import math
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

# Attempt importing joblib
try:
    import joblib
except Exception:
    joblib = None

# Check if scikit-learn and its dependencies (scipy/C-extensions) can be loaded
SKLEARN_AVAILABLE = False
try:
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Scikit-Learn/SciPy could not be loaded due to system policy or missing libraries: {e}")
    print("⚠️ Falling back to pure Python / lightweight model implementation.")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def resolve_path(path):
    """Resolve a file path relative to the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_data(filepath="data/web3_github_market_merged.csv"):
    """Loads preprocessed Web3 dataset."""
    data_path = resolve_path(filepath)

    if not data_path.exists():
        raise FileNotFoundError(
            f"❌ File not found at {data_path}. Run src/02_eda_preprocessing.py first."
        )

    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset with {len(df)} rows.")
    return df


class FallbackLogisticRegression:
    """A lightweight, pure-Python logistic regression classifier used when C-extensions are blocked."""
    def __init__(self, learning_rate=0.01, epochs=500):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.means = {}
        self.stds = {}
        self.feature_importances_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -25, 25)))

    def fit(self, X, y):
        X_mat = X.copy()
        # Handle missing values and scaling
        for col in X_mat.columns:
            mean_val = X_mat[col].median()
            if pd.isna(mean_val):
                mean_val = 0.0
            self.means[col] = mean_val
            X_mat[col] = X_mat[col].fillna(mean_val)
            
            std_val = X_mat[col].std()
            if pd.isna(std_val) or std_val == 0:
                std_val = 1.0
            self.stds[col] = std_val
            X_mat[col] = (X_mat[col] - mean_val) / std_val

        X_arr = X_mat.to_numpy()
        y_arr = y.to_numpy()
        n_samples, n_features = X_arr.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            linear_model = np.dot(X_arr, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X_arr.T, (y_predicted - y_arr))
            db = (1 / n_samples) * np.sum(y_predicted - y_arr)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        abs_w = np.abs(self.weights)
        total_w = np.sum(abs_w) if np.sum(abs_w) > 0 else 1.0
        self.feature_importances_ = abs_w / total_w
        return self

    def predict_proba(self, X):
        X_mat = X.copy()
        for col in X_mat.columns:
            X_mat[col] = X_mat[col].fillna(self.means.get(col, 0.0))
            X_mat[col] = (X_mat[col] - self.means.get(col, 0.0)) / self.stds.get(col, 1.0)
        X_arr = X_mat.to_numpy()
        probs = self._sigmoid(np.dot(X_arr, self.weights) + self.bias)
        return np.column_stack((1 - probs, probs))

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def build_pipeline(feature_cols):
    """Constructs the scikit-learn preprocessing and classification pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    return model_pipeline


def train_and_evaluate(df):
    """Trains the model with cross-validation or fallback classifier."""
    feature_cols = [
        "stars_count",
        "forks_count",
        "open_issues_count",
        "commits_last_30d",
        "price_usd",
        "volume_24h_usd",
        "commits_per_billion_mcap",
        "stars_per_fork",
        "volume_mcap_ratio",
        "log_market_cap",
    ]

    if df.empty:
        raise ValueError("❌ Dataset is empty.")

    missing = [col for col in feature_cols + ["is_high_maturity"] if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    X = df[feature_cols]
    y = df["is_high_maturity"]

    if y.nunique() < 2:
        raise ValueError(
            "❌ Target column 'is_high_maturity' must contain at least two classes for training."
        )

    if SKLEARN_AVAILABLE:
        test_size = 0.25 if len(df) >= 8 else 0.20
        stratify = y if (len(y.unique()) > 1 and min(y.value_counts()) >= 2) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )

        pipeline = build_pipeline(feature_cols)

        param_grid = {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_split": [2, 5],
        }

        min_samples = y_train.value_counts().min()
        if min_samples < 2:
            raise ValueError("❌ At least 2 samples per class are required in the training set for cross-validation.")
        cv_splits = min(3, min_samples)

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_splits,
            scoring="f1_weighted",
            n_jobs=-1,
        )

        print("⏳ Training Random Forest Classifier via GridSearchCV...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"✅ Best Parameters: {grid_search.best_params_}")

        y_pred = best_model.predict(X_test)
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, zero_division=0))

        if len(np.unique(y_test)) > 1:
            y_proba = best_model.predict_proba(X_test)[:, 1]
            try:
                roc_score = roc_auc_score(y_test, y_proba)
                print(f"ROC-AUC Score: {roc_score:.4f}")
            except Exception as e:
                print(f"Could not compute ROC-AUC: {e}")

        classifier_step = best_model.named_steps["classifier"]
        importances = classifier_step.feature_importances_
    else:
        # Fallback execution path
        print("⏳ Training Fallback Pure-Python Logistic Regression...")
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = FallbackLogisticRegression()
        model.fit(X_train, y_train)
        best_model = model

        y_pred = best_model.predict(X_test)
        accuracy = np.mean(y_pred == y_test.to_numpy()) if len(y_test) > 0 else 1.0
        print(f"✅ Fallback Model Trained. Holdout Accuracy: {accuracy:.4f}")
        importances = best_model.feature_importances_

    imp_df = pd.DataFrame(
        {"Feature": feature_cols, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    print("\n--- Feature Importances ---")
    print(imp_df.to_string(index=False))

    return best_model, feature_cols


def save_model(model, filepath="models/web3_maturity_pipeline.pkl"):
    """Saves the trained pipeline using joblib or pickle fallback."""
    model_path = resolve_path(filepath)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if joblib is not None:
        try:
            joblib.dump(model, model_path)
            print(f"\n✅ Model pipeline successfully exported to {model_path}")
            return
        except Exception:
            pass

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ Model pipeline successfully exported to {model_path} via pickle")


def main():
    print("🚀 Starting Machine Learning Training Pipeline...")
    df = load_data()
    best_model, _ = train_and_evaluate(df)
    save_model(best_model)


if __name__ == "__main__":
    main()