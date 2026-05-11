# ================================================================
# PROJECT 2: CUSTOMER CHURN PREDICTION — TELECOM
# Author  : Gaurav Chavan
# Tools   : Python (Pandas, Scikit-learn, SciPy, Matplotlib)
# Dataset : 7,043 telecom customer records with 20 features
# ================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, confusion_matrix)

# ── Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.color':       '#EEEEEE',
    'grid.linewidth':   0.6,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
})

BLUE  = '#185FA5'
GREEN = '#1D9E75'
AMBER = '#BA7517'
RED   = '#A32D2D'
GRAY  = '#AAAAAA'
IMG   = 'images'   # run from project root

# ================================================================
# SECTION 1 — LOAD & INSPECT
# ================================================================
print("=" * 60)
print("SECTION 1: Load & Inspect Data")
print("=" * 60)

df = pd.read_csv('data/telecom_churn.csv')
df['ChurnBin'] = (df['Churn'] == 'Yes').astype(int)

print(f"Shape          : {df.shape}")
print(f"Columns        : {list(df.columns)}")
print(f"\nMissing values : {df.isnull().sum().sum()}")
print(f"Churn rate     : {df['ChurnBin'].mean():.1%}")
print(f"\nChurn counts   :\n{df['Churn'].value_counts()}")
print(f"\nSample rows:")
print(df.head(5).to_string())

# ================================================================
# SECTION 2 — EXPLORATORY DATA ANALYSIS
# ================================================================
print("\n" + "=" * 60)
print("SECTION 2: EDA — Key Churn Drivers")
print("=" * 60)

df['TenureBucket'] = pd.cut(
    df['Tenure'],
    bins=[-1, 12, 24, 48, 72],
    labels=['0-12m', '13-24m', '25-48m', '49-72m'])

# Contract type
ct = df.groupby('Contract')['ChurnBin'].mean() * 100
print(f"\nChurn rate by Contract:\n{ct.round(1).to_string()}")

# Tenure
tb = df.groupby('TenureBucket', observed=True)['ChurnBin'].mean() * 100
print(f"\nChurn rate by Tenure:\n{tb.round(1).to_string()}")

# Internet service
isr = df.groupby('InternetService')['ChurnBin'].mean() * 100
print(f"\nChurn rate by Internet Service:\n{isr.round(1).to_string()}")

# Overtime impact
ot = df.groupby('PaymentMethod')['ChurnBin'].mean() * 100
print(f"\nChurn rate by Payment Method:\n{ot.round(1).to_string()}")

# CHART 1 — Churn Drivers 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(ct.index, ct.values, color=[RED, AMBER, GREEN],
               alpha=0.9, edgecolor='white')
axes[0, 0].set_title('Churn Rate by Contract Type')
axes[0, 0].set_ylabel('Churn Rate (%)')
for i, v in enumerate(ct.values):
    axes[0, 0].text(i, v + 0.4, f'{v:.1f}%',
                    ha='center', fontsize=12, fontweight='bold')

axes[0, 1].bar(tb.index.astype(str), tb.values,
               color=[RED, AMBER, GREEN, BLUE], alpha=0.9, edgecolor='white')
axes[0, 1].set_title('Churn Rate by Customer Tenure')
axes[0, 1].set_ylabel('Churn Rate (%)')
for i, v in enumerate(tb.values):
    axes[0, 1].text(i, v + 0.4, f'{v:.1f}%',
                    ha='center', fontsize=12, fontweight='bold')

axes[1, 0].bar(isr.index, isr.values,
               color=[BLUE, RED, GREEN], alpha=0.9, edgecolor='white')
axes[1, 0].set_title('Churn Rate by Internet Service')
axes[1, 0].set_ylabel('Churn Rate (%)')
for i, v in enumerate(isr.values):
    axes[1, 0].text(i, v + 0.4, f'{v:.1f}%',
                    ha='center', fontsize=12, fontweight='bold')

axes[1, 1].hist(df[df['Churn'] == 'No']['MonthlyCharges'],
                bins=30, alpha=0.6, color=BLUE,
                label='No Churn', density=True)
axes[1, 1].hist(df[df['Churn'] == 'Yes']['MonthlyCharges'],
                bins=30, alpha=0.6, color=RED,
                label='Churned', density=True)
axes[1, 1].axvline(df[df['Churn'] == 'No']['MonthlyCharges'].mean(),
                   color=BLUE, lw=2, linestyle='--')
axes[1, 1].axvline(df[df['Churn'] == 'Yes']['MonthlyCharges'].mean(),
                   color=RED, lw=2, linestyle='--')
axes[1, 1].set_title('Monthly Charges — Churn vs Retained')
axes[1, 1].set_xlabel('Monthly Charges ($)')
axes[1, 1].legend(fontsize=10)

fig.suptitle('Customer Churn Analysis — Key Business Drivers',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{IMG}/01_churn_drivers.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: 01_churn_drivers.png")

# ================================================================
# SECTION 3 — FEATURE ENGINEERING & MODEL TRAINING
# ================================================================
print("\n" + "=" * 60)
print("SECTION 3: Feature Engineering & Model Training")
print("=" * 60)

cat_cols = [
    'Gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'TechSupport', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'TenureBucket'
]

df2 = df.copy()
le = LabelEncoder()
for c in cat_cols:
    df2[c] = le.fit_transform(df2[c].astype(str))

feats = [c for c in df2.columns
         if c not in ['CustomerID', 'Churn', 'ChurnBin']]

X = df2[feats]
y = df2['ChurnBin']

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set : {X_train.shape[0]:,} rows")
print(f"Test set     : {X_test.shape[0]:,} rows")
print(f"Churn in test: {y_test.mean():.1%}")

# Train 3 models
models = [
    ('Logistic Regression',
     LogisticRegression(max_iter=1000, random_state=42,
                        class_weight='balanced')),
    ('Random Forest',
     RandomForestClassifier(n_estimators=100, random_state=42,
                             class_weight='balanced')),
    ('Gradient Boosting',
     GradientBoostingClassifier(n_estimators=100, random_state=42)),
]
model_colors = [BLUE, GREEN, AMBER]
trained_models = {}

print("\nModel Results:")
print(f"{'Model':<25} {'AUC':>6} {'Accuracy':>10}")
print("-" * 45)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for (name, model), col in zip(models, model_colors):
    model.fit(X_train, y_train)
    prob  = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, prob)
    acc   = model.score(X_test, y_test)
    fpr, tpr, _ = roc_curve(y_test, prob)
    trained_models[name] = {'model': model, 'auc': auc, 'prob': prob}

    print(f"{name:<25} {auc:>6.3f} {acc:>10.1%}")
    axes[0].plot(fpr, tpr, color=col, lw=2.5,
                 label=f'{name} (AUC={auc:.3f})')

axes[0].plot([0, 1], [0, 1], '--', color=GRAY, lw=1.5,
             label='Random baseline')
axes[0].fill_between([0, 1], [0, 1], alpha=0.05, color=GRAY)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curves — 3 Model Comparison')
axes[0].legend(fontsize=10)

# Feature Importance from RF
rf_model = trained_models['Random Forest']['model']
imp = pd.Series(rf_model.feature_importances_,
                index=feats).sort_values().tail(12)
bar_colors = [BLUE if v > 0.05 else (GREEN if v > 0.03 else GRAY)
              for v in imp.values]
axes[1].barh(imp.index, imp.values, color=bar_colors, alpha=0.9,
             edgecolor='white')
axes[1].set_title('Top Feature Importances (Random Forest)')
axes[1].set_xlabel('Importance Score')
for i, v in enumerate(imp.values):
    axes[1].text(v + 0.001, i, f'{v:.3f}',
                 va='center', fontsize=9)

fig.suptitle('Churn Prediction — ML Model Comparison',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{IMG}/02_model_roc_features.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("\nSaved: 02_model_roc_features.png")

# ================================================================
# SECTION 4 — BEST MODEL EVALUATION & BUSINESS IMPACT
# ================================================================
print("\n" + "=" * 60)
print("SECTION 4: Best Model Evaluation & Business Impact")
print("=" * 60)

best_model = trained_models['Random Forest']['model']
y_pred     = best_model.predict(X_test)
cm         = confusion_matrix(y_test, y_pred)

print("\nClassification Report — Random Forest:")
print(classification_report(y_test, y_pred,
                             target_names=['Retained', 'Churned']))
print(f"Confusion Matrix:\n{cm}")

# Business calculations
avg_annual_revenue = df['MonthlyCharges'].mean() * 12
total_churners     = int(df['ChurnBin'].sum())
recoverable        = int(total_churners * 0.30)
annual_saved       = recoverable * avg_annual_revenue

print(f"\nBusiness Impact:")
print(f"  Total churners identified : {total_churners:,}")
print(f"  Recoverable (30%)         : {recoverable:,}")
print(f"  Avg annual revenue/cust   : ${avg_annual_revenue:,.0f}")
print(f"  Est. annual revenue saved : ${annual_saved:,.0f}")

# CHART 3 — Confusion Matrix & Business Impact
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

im = axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar(im, ax=axes[0])
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Predicted: No', 'Predicted: Yes'], fontsize=11)
axes[0].set_yticklabels(['Actual: No', 'Actual: Yes'], fontsize=11)
axes[0].set_title('Confusion Matrix — Random Forest')
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f'{cm[i, j]:,}',
                     ha='center', va='center', fontsize=18,
                     fontweight='bold',
                     color='white' if cm[i, j] > thresh else 'black')

cats = ['Total Churners\nIdentified',
        'Recoverable\n(30% targeted)',
        'Estimated\nRetained']
vals = [total_churners, recoverable, int(recoverable * 0.70)]
bar_cols = [RED, AMBER, GREEN]
bars = axes[1].bar(cats, vals, color=bar_cols, alpha=0.9,
                   edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 v + 10, f'{v:,}',
                 ha='center', fontsize=13, fontweight='bold')
axes[1].set_title('Business Impact of Churn Prediction')
axes[1].set_ylabel('Number of Customers')
axes[1].text(
    0.5, -0.18,
    f'Estimated Annual Revenue Saved: ${annual_saved:,.0f}',
    ha='center', transform=axes[1].transAxes,
    fontsize=13, fontweight='bold', color=GREEN,
    bbox=dict(boxstyle='round,pad=0.3',
              facecolor='#EAF3DE', edgecolor='#1D9E75'))

fig.suptitle('Model Evaluation & Revenue Impact',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{IMG}/03_business_impact.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved: 03_business_impact.png")

# ================================================================
# SECTION 5 — KEY FINDINGS SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("SECTION 5: KEY BUSINESS FINDINGS")
print("=" * 60)

best_auc = trained_models['Random Forest']['auc']

print(f"""
KEY FINDINGS — Customer Churn Prediction
──────────────────────────────────────────────────────
Dataset Size          : {len(df):,} customers
Overall Churn Rate    : {df['ChurnBin'].mean():.1%}

TOP CHURN DRIVERS (in order):
  1. Contract type     → Month-to-month: {ct['Month-to-month']:.1f}% churn
  2. Tenure < 12 months → {tb['0-12m']:.1f}% churn rate
  3. Fiber optic internet → {isr['Fiber optic']:.1f}% churn rate
  4. Electronic check payment → highest churn payment method

BEST MODEL: Random Forest
  AUC Score : {best_auc:.3f}
  Accuracy  : {best_model.score(X_test, y_test):.1%}

BUSINESS IMPACT:
  Total churners       : {total_churners:,}
  Recoverable (30%)    : {recoverable:,}
  Annual revenue saved : ${annual_saved:,.0f}

BUSINESS RECOMMENDATIONS:
──────────────────────────────────────────────────────
1. Incentivise month-to-month customers to switch to
   One-year or Two-year contracts with discount offer
2. Target new customers (Tenure < 12m) with onboarding
   programme and check-in calls in first 90 days
3. Investigate Fiber optic service quality issues
4. Migrate Electronic check users to auto-pay methods
""")
print("=" * 60)
print("ALL 3 CHARTS GENERATED — Check images/ folder")
print("=" * 60)
