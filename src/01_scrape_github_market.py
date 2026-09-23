import os
import time
import requests
import pandas as pd

# Optional: Add your GitHub Personal Access Token to avoid strict rate limits (60 req/hr vs 5000 req/hr)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "your_github_token_here")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN != "your_github_token_here" else {}

WEB3_TARGETS = [
    {"symbol": "BTC", "repo_owner": "bitcoin", "repo_name": "bitcoin", "coingecko_id": "bitcoin"},
    {"symbol": "ETH", "repo_owner": "ethereum", "repo_name": "go-ethereum", "coingecko_id": "ethereum"},
    {"symbol": "SOL", "repo_owner": "solana-labs", "repo_name": "solana", "coingecko_id": "solana"},
    {"symbol": "LINK", "repo_owner": "smartcontractkit", "repo_name": "chainlink", "coingecko_id": "chainlink"},
    {"symbol": "AVAX", "repo_owner": "ava-labs", "repo_name": "avalanchego", "coingecko_id": "avalanche-2"},
    {"symbol": "ADA", "repo_owner": "input-output-hk", "repo_name": "cardano-node", "coingecko_id": "cardano"},
    {"symbol": "DOT", "repo_owner": "paritytech", "repo_name": "polkadot-sdk", "coingecko_id": "polkadot"},
    {"symbol": "NEAR", "repo_owner": "near", "repo_name": "nearcore", "coingecko_id": "near"}
]

def fetch_github_metrics(owner, repo):
    """Fetches core repository metadata and recent commit counts."""
    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    try:
        repo_res = requests.get(base_url, headers=HEADERS, timeout=15)
    except requests.exceptions.RequestException as exc:
        print(f"⚠️ Warning: GitHub API request failed for {owner}/{repo}: {exc}")
        return None

    if repo_res.status_code != 200:
        print(f"⚠️ Warning: Could not fetch GitHub repo for {owner}/{repo} (status={repo_res.status_code})")
        return None

    repo_data = repo_res.json()

    # Participation stats endpoint (returns 52 weeks of commit activity)
    try:
        commit_res = requests.get(f"{base_url}/stats/participation", headers=HEADERS, timeout=15)
    except requests.exceptions.RequestException as exc:
        print(f"⚠️ Warning: GitHub commit stats request failed for {owner}/{repo}: {exc}")
        commit_res = None

    commits_last_30d = 0
    if commit_res is not None and commit_res.status_code == 200:
        try:
            commit_data = commit_res.json()
            if "all" in commit_data and len(commit_data["all"]) >= 4:
                commits_last_30d = sum(commit_data["all"][-4:])  # Sum of last 4 weeks
        except ValueError:
            print(f"⚠️ Warning: Invalid JSON returned for commit stats of {owner}/{repo}")

    return {
        "stars_count": repo_data.get("stargazers_count", 0),
        "forks_count": repo_data.get("forks_count", 0),
        "open_issues_count": repo_data.get("open_issues_count", 0),
        "commits_last_30d": commits_last_30d
    }

def fetch_coingecko_data(coingecko_ids):
    """Fetches current market metrics from CoinGecko API."""
    ids_str = ",".join(coingecko_ids)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ids_str,
        "order": "market_cap_desc"
    }

    try:
        res = requests.get(url, params=params, timeout=15)
    except requests.exceptions.RequestException as exc:
        print(f"⚠️ Warning: CoinGecko request failed: {exc}")
        return pd.DataFrame()

    if res.status_code != 200:
        print(f"⚠️ Warning: Failed to retrieve CoinGecko market data (status={res.status_code}).")
        return pd.DataFrame()

    try:
        market_list = res.json()
    except ValueError:
        print("⚠️ Warning: Invalid JSON returned by CoinGecko API.")
        return pd.DataFrame()

    records = []
    for item in market_list:
        records.append({
            "coingecko_id": item.get("id"),
            "price_usd": item.get("current_price", 0.0),
            "market_cap_usd": item.get("market_cap", 0.0),
            "volume_24h_usd": item.get("total_volume", 0.0),
            "price_change_24h_pct": item.get("price_change_percentage_24h", 0.0)
        })
    return pd.DataFrame(records)

def main():
    print("🚀 Starting Web3 Data Scraping Pipeline...")
    
    # 1. Scraping GitHub Activity
    gh_records = []
    for target in WEB3_TARGETS:
        print(f"Fetching GitHub data for {target['symbol']}...")
        gh_data = fetch_github_metrics(target["repo_owner"], target["repo_name"])
        if gh_data:
            gh_data["symbol"] = target["symbol"]
            gh_data["coingecko_id"] = target["coingecko_id"]
            gh_records.append(gh_data)
        time.sleep(1) # Prevent aggressive rate limiting
        
    df_gh = pd.DataFrame(gh_records)
    
    # 2. Fetching CoinGecko Market Metrics
    print("Fetching CoinGecko market data...")
    cg_ids = [t["coingecko_id"] for t in WEB3_TARGETS]
    df_market = fetch_coingecko_data(cg_ids)
    
    # 3. Merge & Save
    if not df_gh.empty and not df_market.empty:
        df_merged = pd.merge(df_gh, df_market, on="coingecko_id", how="inner")
        
        os.makedirs("data", exist_ok=True)
        raw_path = "data/raw_web3_data.csv"
        df_merged.to_csv(raw_path, index=False)
        print(f"✅ Data successfully collected and saved to {raw_path}")
        print(df_merged[["symbol", "commits_last_30d", "price_usd", "market_cap_usd"]])
    else:
        print("❌ Pipeline failed to gather full dataset.")

if __name__ == "__main__":
    main()