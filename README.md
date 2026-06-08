# BTC AI Price Prediction Dashboard

**DFS504 · Domain C — Investment & Market Intelligence · Spring 2026**

Big Data/AI-Powered Bitcoin Price Prediction System
大數據/人工智慧驅動的比特幣價格預測系統

---

## Pages / 頁面

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, key findings, ML framework |
| 📊 Data Explorer | BTC price history, macro correlations, dataset stats |
| 📈 Model Performance | 5-model comparison, CV results, hyperparameters |
| 🔮 Prediction Demo | Input 6-day prices → 7-day forecast |
| 🧠 SHAP Explainability | Feature importance, macro signal interpretation |
| ❤️ Market Health Score | Bull/Bear/Neutral regime detection (VIX, DXY, US10Y, SP500, Gold) |
| 💼 Business Recommendation | User profiles, decision framework, disclaimers |

---

## Local Setup / 本機安裝

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open: http://localhost:8501

---

## Deploy to Streamlit Cloud / 部署到 Streamlit Cloud

1. Push this folder to a **GitHub repository** (public or private)
2. Go to https://share.streamlit.io
3. Click **"New app"**
4. Select your repo, branch `main`, file `app.py`
5. Click **"Deploy"** — get a public URL in ~2 minutes

**Free tier**: sufficient for demo and presentation purposes.

---

## File Structure

```
btc_dashboard/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml         ← Dark theme configuration
└── README.md               ← This file
```

---

## Key Results

| Model | Dataset | Accuracy | MAPE |
|-------|---------|---------|------|
| LightGBM ✅ | BTC Only | **90.40%** | **9.60%** |
| GRU | BTC Only | 96.82%† | 3.18%† |
| XGBoost Tuned | Macro+OC | 96.76%† | 3.24%† |
| LSTM | BTC Only | 89.37% | 10.63% |
| Linear Reg | BTC Only | 98.19%* | 1.81%* |

\* Data leakage. † Different experimental setup.
