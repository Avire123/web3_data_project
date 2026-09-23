
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
