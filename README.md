# ⚡ Web3 Data Project

A full-stack Web3 intelligence project that combines GitHub developer activity with market data to classify project maturity and power an interactive dashboard.

## 🌟 Overview
This project collects and analyzes:
- GitHub repository activity
- Developer metrics like stars, forks, issues, and commits
- Market indicators like price, volume, and market cap
- A machine learning model for maturity prediction

## 🧠 What's Included
- Web scraping pipeline for GitHub and market data
- Preprocessing and feature engineering
- Exploratory data analysis
- Model training and evaluation
- Streamlit dashboard for predictions

## 🗂️ Project Structure
```text
web3_data_project/
├── app.py                          # Streamlit dashboard
├── data/                           # Generated datasets (gitignored)
├── models/                         # Trained model artifacts (gitignored)
├── src/
│   ├── 01_scrape_github_market.py  # Data collection
│   ├── 02_eda_preprocessing.py     # EDA & feature engineering
│   └── 03_train_model.py           # Model training
├── requirements.txt
└── README.md
```

## 🔄 Pipeline Flow
1. 🕷️ Scrape project and market data
2. 🧼 Clean and engineer features
3. 📊 Explore and validate trends
4. 🤖 Train a classification model
5. 💾 Save the model for use in the dashboard

## ▶️ Usage
```powershell
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/01_scrape_github_market.py
python src/02_eda_preprocessing.py
python src/03_train_model.py

# Launch the dashboard
streamlit run app.py
```
