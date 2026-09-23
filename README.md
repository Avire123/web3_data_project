# ⚡ Web3 Data Project

A full-stack Web3 intelligence project that combines GitHub developer activity with market data to classify project maturity and power an interactive dashboard.

## 🌟 Overview
This project collects and analyzes:
- GitHub repository activity
- developer metrics like stars, forks, issues, and commits
- market indicators like price, volume, and market cap
- a machine learning model for maturity prediction

## 🧠 What’s Included
- Web scraping pipeline for GitHub and market data
- preprocessing and feature engineering
- exploratory data analysis
- model training and evaluation
- Streamlit dashboard for predictions

## 🗂️ Project Structure
```text
web3_data_project/
├── app.py
├── data/
├── models/
├── src/
│   ├── 01_scrape_github_market.py
│   ├── 02_eda_preprocessing.py
│   ├── 03_train_model.py
├── README.md
└── requirements.txt
```markdown
# 🧪 Source Pipeline README

This folder contains the project’s data collection, preprocessing, and model training pipeline.

## 📁 Files
- `01_scrape_github_market.py` — gathers GitHub and market data
- `02_eda_preprocessing.py` — cleans, merges, and prepares the dataset
- `03_train_model.py` — trains and saves the maturity model

## 🔄 Pipeline Flow
1. 🕷️ Scrape project and market data
2. 🧼 Clean and engineer features
3. 📊 Explore and validate trends
4. 🤖 Train a classification model
5. 💾 Save the model for use in the dashboard

## ▶️ Usage
```powershell
cd "c:\Users\MCHARO\web3_data_project"
python src\01_scrape_github_market.py
python src\02_eda_preprocessing.py
python src\03_train_model.py
```markdown
# 🧪 Source Pipeline README

This folder contains the project’s data collection, preprocessing, and model training pipeline.

## 📁 Files
- `01_scrape_github_market.py` — gathers GitHub and market data
- `02_eda_preprocessing.py` — cleans, merges, and prepares the dataset
- `03_train_model.py` — trains and saves the maturity model

## 🔄 Pipeline Flow
1. 🕷️ Scrape project and market data
2. 🧼 Clean and engineer features
3. 📊 Explore and validate trends
4. 🤖 Train a classification model
5. 💾 Save the model for use in the dashboard

## ▶️ Usage
```powershell
cd "c:\Users\MCHARO\web3_data_project"
python src\01_scrape_github_market.py
python src\02_eda_preprocessing.py
python src\03_train_model.py
