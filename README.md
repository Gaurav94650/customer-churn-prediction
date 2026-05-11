# Project 2: Customer Churn Prediction — Telecom

**Tools:** Python (Pandas, Scikit-learn, Matplotlib) · SQL  
**Dataset:** 7,043 telecom customer records · 20 features · Churn rate: 32.3%

---

## Business Problem
A telecom company is losing 32.3% of its customers annually. The business needs to:
1. Identify what is driving churn
2. Predict which current customers are about to leave
3. Quantify the revenue at risk and recoverable amount
4. Recommend targeted retention actions

---

## Key Findings

| Metric | Value |
|--------|-------|
| **Total Customers** | 7,043 |
| **Overall Churn Rate** | 32.3% |
| **Best Model** | Random Forest |
| **Model AUC** | 0.624 |
| **Model Accuracy** | 66.8% |
| **Top Churn Driver** | Month-to-month contract (highest churn rate) |
| **Revenue Recoverable** | **$560,114/year** (with 30% retention campaign success) |

---

## Top Churn Drivers (ranked)

1. **Contract Type** — Month-to-month customers churn at 2–3x the rate of annual contract holders
2. **Tenure < 12 months** — Newest customers are most likely to leave
3. **Fiber Optic Internet** — Higher churn than DSL or no internet
4. **Electronic Check payment** — Highest churn rate by payment method

---

## Business Recommendations

1. **Contract conversion:** Offer 10–15% discount to move month-to-month customers to 1-year contracts — projected to retain 200+ customers annually
2. **Onboarding programme:** New customers (Tenure < 12m) need a structured 90-day check-in and value realisation programme
3. **Fiber optic service review:** Investigate quality issues driving higher churn; consider proactive technical outreach to Fiber optic subscribers
4. **Payment migration:** Incentivise Electronic check users to switch to auto-pay (bank transfer or credit card)

---

## ML Models Compared

| Model | AUC | Accuracy |
|-------|-----|----------|
| Logistic Regression | ~0.60 | ~65% |
| **Random Forest** | **0.624** | **66.8%** |
| Gradient Boosting | ~0.61 | ~66% |

---

## Project Structure

```
project2_churn/
├── README.md
├── data/
│   └── telecom_churn.csv          ← 7,043 customer records
├── sql/
│   └── churn_sql_analysis.sql     ← 8 SQL queries (CTEs, window functions)
├── notebooks/
│   └── churn_analysis.py          ← Full Python ML pipeline
└── images/
    ├── 01_churn_drivers.png        ← 2x2 EDA chart
    ├── 02_model_roc_features.png   ← ROC curves + feature importance
    └── 03_business_impact.png      ← Confusion matrix + revenue impact
```

---

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn scipy
cd project2_churn
python notebooks/churn_analysis.py
```

**SQL:**
```
1. Import data/telecom_churn.csv into SQLite as table: customers
2. Run sql/churn_sql_analysis.sql
```

---

## Charts Preview

### Churn Drivers
![Churn Drivers](images/01_churn_drivers.png)

### Model ROC Curves & Feature Importance
![Model Results](images/02_model_roc_features.png)

### Business Impact
![Business Impact](images/03_business_impact.png)

---

## Skills Demonstrated

- **Python ML:** Logistic Regression, Random Forest, Gradient Boosting comparison
- **Evaluation:** ROC-AUC, Confusion Matrix, Classification Report
- **Feature Engineering:** Label encoding, StandardScaler, tenure bucketing
- **SQL:** CTEs, CASE WHEN bucketing, RANK window function, subqueries
- **Business Thinking:** Revenue impact quantification, actionable retention strategy

---

## Resume Bullet

> *Built customer churn prediction model using Python (Scikit-learn) on 7,043 telecom records — compared Logistic Regression, Random Forest & Gradient Boosting; best model achieved AUC of 0.624; identified month-to-month contracts as top churn driver and quantified $560K in recoverable annual revenue through targeted retention strategy.*
