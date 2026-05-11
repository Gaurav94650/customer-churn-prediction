-- ================================================================
-- PROJECT 2: CUSTOMER CHURN ANALYSIS — SQL QUERIES
-- Author  : [Your Name]
-- Tool    : SQLite or MySQL
-- Dataset : telecom_churn.csv → import as table named: customers
-- ================================================================


-- ──────────────────────────────────────────────────────────────
-- QUERY 1: Overall Churn Rate
-- Business: What % of our customers are churning?
-- ──────────────────────────────────────────────────────────────
SELECT
    Churn,
    COUNT(*)  AS customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 1) AS percentage
FROM customers
GROUP BY Churn;


-- ──────────────────────────────────────────────────────────────
-- QUERY 2: Churn Rate by Contract Type
-- Business: Which contract type has the highest churn?
-- ──────────────────────────────────────────────────────────────
SELECT
    Contract,
    COUNT(*)  AS total_customers,
    SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) * 100.0
          / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- ──────────────────────────────────────────────────────────────
-- QUERY 3: Churn Rate by Internet Service Type
-- Business: Is Fiber optic driving churn?
-- ──────────────────────────────────────────────────────────────
SELECT
    InternetService,
    COUNT(*)  AS total_customers,
    ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) * 100.0
          / COUNT(*), 1) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- ──────────────────────────────────────────────────────────────
-- QUERY 4: Revenue Comparison — Churned vs Retained
-- Business: What revenue are we losing to churn?
-- ──────────────────────────────────────────────────────────────
SELECT
    Churn,
    COUNT(*)                              AS customers,
    ROUND(AVG(MonthlyCharges), 2)         AS avg_monthly_charges,
    ROUND(AVG(TotalCharges), 2)           AS avg_total_charges,
    ROUND(SUM(MonthlyCharges), 0)         AS total_monthly_revenue,
    ROUND(AVG(Tenure), 1)                 AS avg_tenure_months
FROM customers
GROUP BY Churn;


-- ──────────────────────────────────────────────────────────────
-- QUERY 5: High-Value Customers at Risk
-- Business: Which high-paying customers are month-to-month AND new?
-- ──────────────────────────────────────────────────────────────
SELECT
    CustomerID,
    Contract,
    InternetService,
    MonthlyCharges,
    Tenure,
    PaymentMethod
FROM customers
WHERE
    Churn       = 'No'
    AND Contract    = 'Month-to-month'
    AND MonthlyCharges > 70
    AND Tenure      < 12
ORDER BY MonthlyCharges DESC
LIMIT 20;


-- ──────────────────────────────────────────────────────────────
-- QUERY 6: Churn Rate by Payment Method (WINDOW FUNCTION)
-- Business: Which payment methods correlate with churn?
-- Skill: Window function (RANK)
-- ──────────────────────────────────────────────────────────────
SELECT
    PaymentMethod,
    total_customers,
    churn_rate_pct,
    RANK() OVER (ORDER BY churn_rate_pct DESC) AS churn_risk_rank
FROM (
    SELECT
        PaymentMethod,
        COUNT(*) AS total_customers,
        ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) * 100.0
              / COUNT(*), 1) AS churn_rate_pct
    FROM customers
    GROUP BY PaymentMethod
)
ORDER BY churn_risk_rank;


-- ──────────────────────────────────────────────────────────────
-- QUERY 7: Tenure Bucket Churn Analysis (CTE)
-- Business: Do new customers churn more than long-term ones?
-- Skill: CTE, CASE WHEN bucketing
-- ──────────────────────────────────────────────────────────────
WITH tenure_buckets AS (
    SELECT
        CustomerID,
        Churn,
        MonthlyCharges,
        CASE
            WHEN Tenure BETWEEN 0  AND 12 THEN '0-12 months'
            WHEN Tenure BETWEEN 13 AND 24 THEN '13-24 months'
            WHEN Tenure BETWEEN 25 AND 48 THEN '25-48 months'
            ELSE '49+ months'
        END AS tenure_group
    FROM customers
)
SELECT
    tenure_group,
    COUNT(*)  AS customers,
    SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) * 100.0
          / COUNT(*), 1) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM tenure_buckets
GROUP BY tenure_group
ORDER BY churn_rate_pct DESC;


-- ──────────────────────────────────────────────────────────────
-- QUERY 8: Senior Citizens vs Non-Senior Churn
-- Business: Do senior citizens have higher churn risk?
-- ──────────────────────────────────────────────────────────────
SELECT
    CASE WHEN SeniorCitizen=1 THEN 'Senior Citizen' ELSE 'Non-Senior' END AS segment,
    COUNT(*)  AS total,
    ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) * 100.0
          / COUNT(*), 1) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY SeniorCitizen;

-- ================================================================
-- END OF QUERIES — 8 business questions answered
-- ================================================================
