# 🎮 Steam KPI Intelligence (SKI)
### Predictive Analytics & Lifecycle Management for Steam Game Success

> *"Predicting what makes a game win, before it launches."*

---

## 🧩 Overview

**Steam KPI Intelligence (SKI)** is a complete **data science and MLOps framework** that predicts the commercial performance of games on the Steam platform.  
It unites data engineering, machine learning, explainable AI, and web deployment into one cohesive analytical system.

The goal is simple: **forecast player adoption and success probability** from pricing, engagement, and sentiment metrics — and visualize the results interactively.

---

## 🚀 Key Objectives

- Predict **game ownership (log-scaled)** using regression models  
- Classify **top-decile success probability** using ensemble learning  
- Explain **model behavior** using SHAP, LIME, and permutation importance  
- Deploy an interactive **Dash web app** for real-time scenario testing  
- Automate **data drift detection** and **model retraining**

---

## 📊 Problem Statement

Game studios, publishers, and investors face uncertainty when estimating how a game will perform after release.  
SKI addresses this by using historical Steam data to **quantify the relationship between pricing, engagement, and success** — giving decision-makers predictive insight.

---

## 🧠 Methodology

### 1️⃣ Data Wrangling & Feature Engineering

Processed and standardized **86,500+ Steam game records**.  
Created derived variables capturing pricing, ownership, engagement, and sentiment behavior.

Key engineered features include:
- `owners_mid`, `votes_total`, `sentiment_ratio`
- `price_delta`, `engagement_rate_2w`, `price_per_hour`

Each numeric field was normalized and validated to remove outliers and ensure schema consistency.

---

### 2️⃣ Modeling Framework

- **Regression Target:** `log(owners_mid)`  
- **Classification Target:** Binary indicator for top-decile ownership  
- **Algorithms Used:**
  - `ElasticNetCV`
  - `GradientBoostingRegressor`
  - `LogisticRegressionCV`
  - `GradientBoostingClassifier`

Feature selection applied:
- Variance thresholding  
- Correlation pruning  
- L1-regularization (LASSO)

---

### 3️⃣ Explainability & Interpretability

- **Permutation Importance:** Global feature contribution  
- **SHAP & LIME:** Local interpretability  
- **Decision Tree Surrogates:** Approximation for decision transparency  

**Top predictive drivers:**
- `votes_total` (review count)  
- `price_delta` (discount magnitude)  
- `average_2weeks_hours` (recent engagement)  
- `average_forever_hours` (lifetime engagement)

---

### 4️⃣ Interactive Dash Deployment

Deployed a responsive **Dash** web app for simulation and reporting.

**App Tabs:**
- 🎯 *Single Scenario* — Interactive prediction  
- 📁 *Batch Scoring* — Bulk CSV scoring  
- 🔍 *Insights* — Model transparency and metadata display  

Includes a **Price × Discount Heatmap** for revenue optimization analysis.

---

### 5️⃣ Drift Monitoring & Retraining

- Tracks data drift via feature distribution divergence  
- Auto-rebuilds engineered features and targets on detection  
- Saves model versions in `/artifacts/model_history`  
- Logs retraining metadata in `heartbeat_log.csv`

---

### 6️⃣ Lifecycle Automation

**APScheduler** automates:
- Metric logging  
- Drift checks  
- Retraining triggers  

Regression metrics: RMSE, MAE, R²  
Classification metrics: ROC-AUC, F1, Brier Score

---

## 📈 Results

| Task | Model | Metric | Value |
|------|--------|--------|--------|
| Regression | GradientBoostingRegressor | RMSE | **1.188** |
| Regression | ElasticNetCV | R² | **0.081** |
| Classification | GradientBoostingClassifier | ROC-AUC | **0.944** |
| Classification | GradientBoostingClassifier | Accuracy | **0.953** |
| Classification | LogisticCV | F1-Score | **0.731** |

**Interpretation:**  
Moderate discounts and strong engagement metrics correlate with ownership growth, while excessive discounts reduce revenue efficiency.

---

## ⚙️ Technical Stack

- **Language:** Python 3.12  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `shap`, `lime`, `plotly`, `dash`, `joblib`, `apscheduler`  
- **Environment:** Local + deployable on Render or Heroku  
- **Version Control:** Git with timestamped model artifacts  
- **MLOps Components:** Drift detection, retraining, explainability, web deployment

---

# 🖥️ How to Use the Dashboard

Once launched, open your browser at:  
👉 **http://127.0.0.1:8051**

The dashboard allows interactive simulation, bulk scoring, and interpretability review.  
It contains three main tabs: **Single Scenario**, **Batch Scoring**, and **Insights**.

---

### 🎯 Single Scenario Tab

Simulate the launch of a single game by entering its KPIs.  
Each field corresponds to a measurable feature from the dataset.

| Feature | Description | Typical Range |
|----------|--------------|----------------|
| `price` | Current sale price (in cents) | 0 – 10,000 |
| `initialprice` | Original price before discounts | 0 – 10,000 |
| `discount` | Discount percentage | 0 – 100 % |
| `userscore` | Steam user score | 0 – 100 |
| `votes_total` | Total reviews (positive + negative) | 0 – 1,000,000+ |
| `sentiment` | Positive review ratio | 0 – 1 |
| `ccu` | Concurrent users (current) | 0 – 100,000+ |
| `average_forever_hours` | Mean lifetime playtime | 0 – 100+ |
| `average_2weeks_hours` | Mean recent playtime | 0 – 50+ |
| `median_forever_hours` | Median lifetime playtime | 0 – 100+ |
| `price_delta` | Difference between original and sale price | Variable |
| `price_per_hour` | Price divided by playtime | 0 – 5,000 |
| `engagement_rate_2w` | Active share of players (2 weeks) | 0 – 1 |

After entering values, click **“Predict Single Scenario.”**

**Outputs:**

- **Owners (pred):** Predicted number of game owners  
- **Success Prob:** Probability of being in the top 10 % of games  
- **Decision Badge:** PASS if above classification threshold, else FAIL  

---

#### 🔥 Price × Discount Heatmap

Visualizes how **price** and **discount** jointly affect projected revenue (`price × predicted owners`).

- **X-Axis:** Discount percentage  
- **Y-Axis:** Price  
- **Color Gradient:** Brighter = higher expected revenue  

**Interpretation:**

- Bright zones identify optimal price-discount combinations  
- Moderate discounts at mid-range prices often yield peak performance  
- Excessive discounts (>80 %) reduce total revenue despite volume gains  

---

### 📁 Batch Scoring Tab

Score multiple titles simultaneously — ideal for publishers and analysts.

**Steps:**

1. Upload a `.csv` file containing all model features.  
2. Click **“Score File.”**  
3. The app:
   - Validates and reorders feature columns  
   - Applies consistent preprocessing  
   - Runs both regression and classification models  
   - Computes a `revenue_proxy` metric  
4. Preview top 200 records directly in-app  
5. Use **“Download Predictions”** to export full results  

**Output Columns:**

| Column | Meaning |
|---------|----------|
| `owners_pred` | Predicted total owners |
| `success_prob` | Probability of top-decile success |
| `revenue_proxy` | Price × predicted owners |
| _Original features_ | Retained for auditing and analysis |

---

### 🔍 Insights Tab

Provides transparency and governance for deployed models.

**Sections:**

1. **Global Importance (Regression):**  
   - Top 15 predictors of ownership (`y_reg_log_owners`)
2. **Global Importance (Classification):**  
   - Top 15 predictors of success probability (`y_clf_success_top10`)
3. **Deployed Thresholds:**  
   - JSON metadata including cutoff probabilities, feature quantiles, and version info

**Purpose:**  
Understand which KPIs drive model outcomes and maintain trust in deployed predictions.

---

### 💡 Practical Use Cases

- **Developers:** Optimize pricing and promotion strategy  
- **Publishers:** Evaluate portfolio performance  
- **Investors:** Forecast ROI potential  
- **Analysts:** Monitor engagement and market sentiment trends  

---

### 🧭 Recommended Workflow

1. Explore **Single Scenario** for hypothesis testing  
2. Use **Batch Scoring** for large-scale evaluation  
3. Review **Insights** for interpretability and model audit  
4. Export results for integration into internal BI pipelines  

---

## 🧩 Outputs

| Artifact | Description |
|-----------|--------------|
| `model_reg.pkl`, `model_clf.pkl` | Trained model artifacts |
| `importance_reg_permutation.csv` | Feature importance results |
| `app_dash.py` | Dash web app |
| `drift_report.csv`, `heartbeat_log.csv` | Drift and monitoring logs |
| `model_history/` | Versioned retraining archive |

---

## 🧭 Impact

**Steam KPI Intelligence (SKI)** demonstrates a full machine-learning lifecycle — from exploration to automated deployment — highlighting mastery in:

- Quantitative modeling  
- Data science engineering  
- MLOps automation  
- Model interpretability  
- Full-stack deployment with Dash  

---

## 👨‍💻 Developer

**Russell I. Lancaster**  
_MSc in Financial Engineering | Data Scientist_  
**Waukesha, Wisconsin**

📧 `Russell.Lancaster243@gmail.com`  
🔗 [LinkedIn: Russell Lancaster](https://www.linkedin.com/in/rlancaster243)

---

