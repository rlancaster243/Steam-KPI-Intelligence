# 🎮 Steam KPI Intelligence (SKI)
### A Predictive Analytics & Lifecycle Management Framework for Steam Game Success

> *"Predicting what makes a game win, before it launches."*

---

## 🧩 Overview
**Steam KPI Intelligence (SKI)** is a full-stack **data science and MLOps framework** designed to forecast the commercial success of games on the Steam platform.  

It combines **feature-rich data engineering**, **machine learning modeling**, **explainable AI**, and **automated retraining pipelines** into one production-ready analytics system.  

Built for both **quantitative insight** and **operational deployment**, SKI models and visualizes how pricing, engagement, and sentiment metrics drive player adoption and success probability.

---

## 🚀 Key Objectives
- Predict **game ownership (log-scaled)** using regression modeling  
- Classify **top-decile commercial success** using ensemble classification  
- Quantify **feature impact and interpretability** via SHAP, LIME, and permutation analysis  
- Deploy an interactive **Dash web app** for scenario testing and pricing simulation  
- Implement **data drift detection**, **retraining**, and **lifecycle monitoring**

---

## 📊 Problem Statement
Developers and investors face uncertainty when estimating how new games will perform on digital platforms.  
SKI provides **predictive intelligence** by learning from historical Steam marketplace data to model ownership, engagement, and revenue potential across diverse genres and pricing structures.

---

## 🧠 Methodology

### **1. Data Wrangling & Feature Engineering**
Cleaned and standardized >86,000 game records from the Steam dataset.  
Key derived features:
- `owners_mid`, `votes_total`, `sentiment_ratio`
- `price_delta`, `engagement_rate_2w`, `price_per_hour`

Applied normalization, outlier removal, and consistency checks across all numeric variables.

---

### **2. Modeling Framework**
- **Regression Target:** Log of median ownership (`log(owners_mid)`)
- **Classification Target:** Binary success indicator (top 10% ownership)
- **Algorithms:**
  - `ElasticNetCV`, `GradientBoostingRegressor`
  - `LogisticRegressionCV`, `GradientBoostingClassifier`
- Feature selection via **variance thresholding**, **correlation pruning**, and **L1 regularization**

---

### **3. Explainability & Interpretability**
- **Permutation Importance:** Global feature relevance  
- **SHAP & LIME:** Local contribution and contrastive examples  
- **Decision Tree Surrogates:** Rule-based model approximations  

**Top Predictors:**
- Review count (`votes_total`)  
- Price × Discount interaction  
- Average 2-week playtime  
- Lifetime playtime per user  

---

### **4. Interactive Dash Deployment**
Deployed a lightweight web dashboard for real-time scenario analysis.

**App Tabs:**
- 🎯 *Single Scenario:* Predict success interactively with sliders and inputs  
- 📁 *Batch Scoring:* Upload CSVs for mass predictions  
- 🔍 *Insights:* Visualize feature importances and KPI thresholds  

Includes a dynamic **Price × Discount Heatmap** to visualize optimal revenue zones.

---

### **5. Drift Monitoring & Retraining**
- **Drift detection** using statistical divergence on feature distributions  
- **Retraining pipeline** auto-rebuilds engineered features and target columns  
- Versioned models saved under `artifacts/model_history/`  
- **Flask health endpoint** reports model freshness and performance diagnostics  

---

### **6. Lifecycle Automation**
Automated monitoring via **APScheduler** logs metrics and health states:
- Regression: RMSE, MAE, R²  
- Classification: ROC-AUC, F1, Brier Score  
- Drift and retraining events tracked in `heartbeat_log.csv`  

---

## 📈 Results

| Task | Model | Metric | Value |
|------|--------|--------|--------|
| Regression | GradientBoostingRegressor | RMSE | **1.188** |
| Regression | ElasticNetCV | R² | **0.081** |
| Classification | GradientBoostingClassifier | ROC-AUC | **0.944** |
| Classification | GradientBoostingClassifier | Accuracy | **0.953** |
| Classification | LogisticCV | F1-score | **0.731** |

**Interpretation:**  
Pricing, engagement, and review volume are the strongest predictors of ownership and success.  
Discounts boost adoption up to a saturation point, with diminishing marginal gains beyond ~60%.

---

## ⚙️ Technical Stack
- **Language:** Python 3.12  
- **Libraries:** pandas, numpy, scikit-learn, shap, lime, joblib, plotly, dash, APScheduler  
- **Environment:** Local + deployable on Render / Heroku  
- **Version Control:** Git + timestamped model registry  
- **MLOps Components:** Drift detection, retraining, explainability, live deployment

---

