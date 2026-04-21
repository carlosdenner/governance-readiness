import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Data Loaded. Shape: {eo_df.shape}")

# --- Step 2: Calculate Tier 1 Score ---
# Using inspected columns: 
# 34_data_docs (Documentation)
# 40_has_ato (Authorization)
# 59_ai_notice (Transparency/Notice)

def score_docs(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    # If it explicitly says missing or not available, score 0
    if 'missing' in s or 'not available' in s:
        return 0
    # If it says complete, available, existing, partially -> 1
    if 'complete' in s or 'available' in s or 'exist' in s or 'partially' in s:
        return 1
    return 0

def score_ato(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    if s.startswith('yes'):
        return 1
    return 0

def score_notice(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    # If none or n/a -> 0
    if 'none' in s or 'n/a' in s:
        return 0
    # If online, terms, physical -> 1
    if 'online' in s or 'physical' in s or 'terms' in s or 'instruction' in s:
        return 1
    return 0

# Apply scoring
eo_df['score_docs'] = eo_df['34_data_docs'].apply(score_docs)
eo_df['score_ato'] = eo_df['40_has_ato'].apply(score_ato)
eo_df['score_notice'] = eo_df['59_ai_notice'].apply(score_notice)

eo_df['tier1_score'] = eo_df['score_docs'] + eo_df['score_ato'] + eo_df['score_notice']

# --- Step 3: Split into Groups ---
median_score = eo_df['tier1_score'].median()
print(f"\nTier 1 Score Distribution:\n{eo_df['tier1_score'].value_counts().sort_index()}")
print(f"Median Score: {median_score}")

# High > Median, Low <= Median
# If median is high (e.g. 2 or 3), this split might be unbalanced, but we stick to the plan.
eo_df['governance_group'] = np.where(eo_df['tier1_score'] > median_score, 'High Governance', 'Low Governance')
print(f"\nGroup Counts:\n{eo_df['governance_group'].value_counts()}")

# --- Step 4: Analyze Code Access ---
# Target: 38_code_access
# We want 'publicly available'.
def score_public_code(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    if 'publicly available' in s:
        return 1
    return 0

eo_df['is_public_code'] = eo_df['38_code_access'].apply(score_public_code)

# Contingency Table
contingency = pd.crosstab(eo_df['governance_group'], eo_df['is_public_code'])
contingency.columns = ['Not Public', 'Public']

print("\nContingency Table (Public Code Access):")
print(contingency)

# Chi-square
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Plot
props = contingency.div(contingency.sum(axis=1), axis=0)
ax = props.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'])
plt.title(f'Public Code Access by Governance Tier (Median={median_score})')
plt.ylabel('Proportion')
plt.xlabel('Governance Group')
plt.legend(title='Code Status', loc='upper right')
plt.tight_layout()
plt.show()