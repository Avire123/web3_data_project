import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Attempt importing joblib
try:
    import joblib
except Exception:
    joblib = None


# Define FallbackLogisticRegression class so pickle can locate it during unpickling
class FallbackLogisticRegression:
    """Fallback classifier wrapper matching the class name from script 03."""

    def __init__(self, weights, intercept, feature_names=None):
        self.weights = np.array(weights)
        self.intercept = float(intercept)
        self.feature_names = feature_names or []
        self.coef_ = self.weights.reshape(1, -1)

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X_mat = X.values
        else:
            X_mat = np.array(X)

        logits = np.dot(X_mat, self.weights) + self.intercept
        probs_class_1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        probs_class_0 = 1.0 - probs_class_1
        return np.column_stack([probs_class_0, probs_class_1])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


# Page configuration
st.set_page_config(
    page_title="Web3 Developer & Market Intelligence",
    page_icon="⚡",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }
    .stMetric { background-color: #1E222D; padding: 15px; border-radius: 8px; }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    data_path = "data/web3_github_market_merged.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


@st.cache_resource
def load_model():
    model_path = "models/web3_maturity_pipeline.pkl"
    if os.path.exists(model_path):
        if joblib is not None:
            try:
                return joblib.load(model_path)
            except Exception:
                pass
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


def main():
    st.title("⚡ Web3 Developer & Market Intelligence Dashboard")
    st.markdown(
        "End-to-end analytics combining off-chain **GitHub activity** with on-chain **CoinGecko market metrics**."
    )

    df = load_data()
    model = load_model()

    if df is None or model is None:
        st.error(
            "⚠️ Dataset or model not found. Please ensure you have run scripts 01, 02, and 03 in `src/` first."
        )
        st.info(
            "Run these commands in your VS Code terminal:\n"
            "`python src/01_scrape_github_market.py`\n"
            "`python src/02_eda_preprocessing.py`\n"
            "`python src/03_train_model.py`"
        )
        return

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs(
        ["📊 Ecosystem EDA", "🤖 Model Performance", "🔮 Real-time Predictor"]
    )

    # --- TAB 1: EDA ---
    with tab1:
        st.header("Web3 Token & Developer Overview")

        # KPI Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Web3 Tokens", len(df))
        col2.metric("Avg 30d Commits", f"{int(df['commits_last_30d'].mean())}")
        col3.metric("Avg Price (USD)", f"${df['price_usd'].mean():,.2f}")
        col4.metric(
            "Avg Market Cap", f"${df['market_cap_usd'].mean() / 1e9:.2f}B"
        )

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("GitHub Activity vs Market Cap")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.scatterplot(
                data=df,
                x="commits_last_30d",
                y="market_cap_usd",
                hue="is_high_maturity",
                style="symbol",
                s=150,
                palette="viridis",
                ax=ax,
            )
            ax.set_yscale("log")
            ax.set_title("Commits (30d) vs. Market Cap (Log Scale)")
            ax.set_xlabel("30-Day Commit Count")
            ax.set_ylabel("Market Cap (USD, Log)")
            st.pyplot(fig)

        with col_right:
            st.subheader("Market Metrics Table")
            display_cols = [
                "symbol",
                "commits_last_30d",
                "stars_count",
                "price_usd",
                "market_cap_usd",
                "commits_per_billion_mcap",
            ]
            st.dataframe(
                df[display_cols].sort_values(
                    by="market_cap_usd", ascending=False
                ),
                use_container_width=True,
            )

    # --- TAB 2: MODEL PERFORMANCE ---
    with tab2:
        st.header("Model Feature Importance Analysis")

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

        # Extract Feature Importances gracefully across different model types
        target_model = (
            model.named_steps["classifier"]
            if hasattr(model, "named_steps")
            else model
        )

        if hasattr(target_model, "feature_importances_"):
            importances = target_model.feature_importances_
        elif hasattr(target_model, "coef_"):
            # Normalize logistic regression absolute coefficients as proxy importance
            coefs = np.abs(target_model.coef_.ravel())
            importances = coefs / (coefs.sum() + 1e-9)
        elif hasattr(target_model, "weights"):
            weights = np.abs(np.array(target_model.weights))
            importances = weights / (weights.sum() + 1e-9)
        else:
            importances = np.ones(len(feature_cols)) / len(feature_cols)

        imp_df = pd.DataFrame(
            {"Feature": feature_cols, "Importance": importances}
        ).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(imp_df["Feature"], imp_df["Importance"], color="#4F46E5")
        ax.set_title("Model Feature Importances")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)

    # --- TAB 3: INFERENCE FORM ---
    with tab3:
        st.header("Predict Token Maturity Stage")
        st.markdown(
            "Adjust off-chain developer metrics and on-chain market metrics to run real-time inference."
        )

        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("💻 Off-Chain Developer Activity")
                stars = st.number_input(
                    "GitHub Stars Count", min_value=0, value=12000, step=500
                )
                forks = st.number_input(
                    "GitHub Forks Count", min_value=0, value=3000, step=100
                )
                open_issues = st.number_input(
                    "Open Issues Count", min_value=0, value=250, step=10
                )
                commits = st.number_input(
                    "Commits (Last 30 Days)", min_value=0, value=180, step=10
                )

            with col_b:
                st.subheader("📈 On-Chain Market Metrics")
                price = st.number_input(
                    "Token Price (USD)", min_value=0.0, value=25.0, step=1.0
                )
                volume_24h = st.number_input(
                    "24h Trading Volume (USD)",
                    min_value=0.0,
                    value=500_000_000.0,
                    step=10_000_000.0,
                )
                market_cap = st.number_input(
                    "Market Capitalization (USD)",
                    min_value=0.0,
                    value=4_000_000_000.0,
                    step=100_000_000.0,
                )

            submit_button = st.form_submit_button("⚡ Run Maturity Prediction")

        if submit_button:
            # Feature Engineering on user input
            commits_per_billion = commits / (market_cap / 1e9 + 1e-5)
            stars_per_fork = stars / (forks + 1)
            volume_mcap_ratio = volume_24h / (market_cap + 1e-5)
            log_mcap = np.log1p(market_cap)

            input_dict = {
                "stars_count": [stars],
                "forks_count": [forks],
                "open_issues_count": [open_issues],
                "commits_last_30d": [commits],
                "price_usd": [price],
                "volume_24h_usd": [volume_24h],
                "commits_per_billion_mcap": [commits_per_billion],
                "stars_per_fork": [stars_per_fork],
                "volume_mcap_ratio": [volume_mcap_ratio],
                "log_market_cap": [log_mcap],
            }

            input_df = pd.DataFrame(input_dict)

            prediction = int(model.predict(input_df)[0])
            probabilities = model.predict_proba(input_df)[0]
            confidence = probabilities[prediction] * 100

            st.markdown("---")
            if prediction == 1:
                st.success(
                    f"🏆 **Prediction: High Maturity / Institutional Grade** (Confidence: {confidence:.1f}%)"
                )
                st.write(
                    "This project shows strong developer commitment relative to its market valuation and healthy trading volume."
                )
            else:
                st.warning(
                    f"⚠️ **Prediction: Early Stage / Speculative** (Confidence: {confidence:.1f}%)"
                )
                st.write(
                    "This project exhibits characteristics typical of early-stage or higher-risk Web3 tokens."
                )


if __name__ == "__main__":
    main()