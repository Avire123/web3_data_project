import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_raw_data(filepath="data/raw_web3_data.csv"):
    """Loads raw scraped Web3 data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Could not find {filepath}. Run src/01_scrape_github_market.py first.")
    
    df = pd.read_csv(filepath)
    print(f"✅ Raw dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def engineer_features(df):
    """Computes cross-domain ratio features and defines classification targets."""
    df = df.copy()

    # Fill missing values if any
    numeric_cols = ["stars_count", "forks_count", "open_issues_count", "commits_last_30d", 
                    "price_usd", "market_cap_usd", "volume_24h_usd"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 1. Developer Intensity Index: Commits per $1 Billion Market Cap
    df["commits_per_billion_mcap"] = df["commits_last_30d"] / (df["market_cap_usd"] / 1e9 + 1e-5)

    # 2. Community Engagement Ratio: Stars per Fork
    df["stars_per_fork"] = df["stars_count"] / (df["forks_count"] + 1)

    # 3. Market Liquidity Ratio: 24h Volume to Market Cap
    df["volume_mcap_ratio"] = df["volume_24h_usd"] / (df["market_cap_usd"] + 1e-5)

    # 4. Log-transformed Skewed Features
    df["log_market_cap"] = np.log1p(df["market_cap_usd"])
    df["log_commits"] = np.log1p(df["commits_last_30d"])
    df["log_volume"] = np.log1p(df["volume_24h_usd"])

    # 5. Define Classification Target: High Maturity / Institutional Grade (1) vs Early / Speculative (0)
    commit_median = df["commits_last_30d"].median()
    mcap_median = df["market_cap_usd"].median()

    df["is_high_maturity"] = (
        (df["commits_last_30d"] >= commit_median) & 
        (df["market_cap_usd"] >= mcap_median)
    ).astype(int)

    print("✅ Feature engineering completed.")
    return df


def generate_eda_plots(df):
    """Generates and saves exploratory data analysis plots."""
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    
    # 1. GitHub Activity vs Market Cap Scatter Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df, 
        x="commits_last_30d", 
        y="market_cap_usd", 
        hue="is_high_maturity", 
        palette="viridis", 
        s=120, 
        style="symbol"
    )
    plt.yscale("log")
    plt.title("30-Day GitHub Commits vs. Market Capitalization (Log Scale)")
    plt.xlabel("30-Day Commit Count")
    plt.ylabel("Market Cap (USD, Log Scale)")
    plt.tight_layout()
    plt.show()

    # 2. Correlation Heatmap
    plt.figure(figsize=(9, 6))
    corr_cols = [
        "stars_count", "forks_count", "commits_last_30d", 
        "price_usd", "market_cap_usd", "volume_24h_usd",
        "commits_per_billion_mcap", "stars_per_fork", "volume_mcap_ratio"
    ]
    corr_matrix = df[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Web3 Off-Chain & On-Chain Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def main():
    print("🚀 Starting EDA & Preprocessing Pipeline...")
    
    # Load
    raw_df = load_raw_data()
    
    # Process
    processed_df = engineer_features(raw_df)
    
    # Save processed data
    output_path = "data/web3_github_market_merged.csv"
    processed_df.to_csv(output_path, index=False)
    print(f"✅ Processed dataset saved to {output_path}")

    # Visualize
    generate_eda_plots(processed_df)


if __name__ == "__main__":
    main()