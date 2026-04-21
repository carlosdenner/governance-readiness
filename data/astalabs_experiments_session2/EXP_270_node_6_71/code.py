import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
import re

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Target columns
impact_col = '17_impact_type'
mitigation_col = '62_disparity_mitigation'

# Drop rows where impact type is missing
eo_data = eo_data.dropna(subset=[impact_col])

# ---------------------------------------------------------
# 1. Categorize Systems (Safety-Only vs Rights-Related)
# ---------------------------------------------------------
# Normalize text
eo_data[impact_col] = eo_data[impact_col].astype(str)

# Define masks
is_rights = eo_data[impact_col].str.contains('Rights', case=False, na=False)
is_safety = eo_data[impact_col].str.contains('Safety', case=False, na=False)

# Assign groups
# Priority: If it has Rights (even if mixed with Safety), it goes to 'Rights-Related'.
# If it has Safety but NO Rights, it goes to 'Safety-Only'.
conditions = [
    is_rights,
    (is_safety & ~is_rights)
]
choices = ['Rights-Related', 'Safety-Only']
eo_data['impact_group'] = np.select(conditions, choices, default='Other')

# Filter for only the groups of interest
analysis_df = eo_data[eo_data['impact_group'].isin(['Rights-Related', 'Safety-Only'])].copy()

print(f"Data filtered. Rows: {len(analysis_df)}")
print(f"Counts by group:\n{analysis_df['impact_group'].value_counts()}")

# ---------------------------------------------------------
# 2. Parse Outcome Variable (Disparity Mitigation) using Heuristics
# ---------------------------------------------------------

def classify_mitigation(text):
    if pd.isna(text):
        return 0 # Treat missing as no evidence of mitigation
    
    text = str(text).lower().strip()
    
    # Negative indicators (Strong override)
    # "no analysis", "none", "n/a", "not applicable", "no specific", "does not"
    negatives = [
        r'no\s+analysis', r'no\s+specific', r'^none', r'n/a', r'not\s+applicable', 
        r'does\s+not', r'no\s+impact', r'no\s+mitigation'
    ]
    for neg in negatives:
        if re.search(neg, text):
            return 0
            
    # Positive indicators
    # "test", "eval", "monitor", "review", "audit", "assess", "mitigat", "bias", "fair", "check"
    positives = [
        'test', 'eval', 'monitor', 'review', 'audit', 'assess', 
        'mitigat', 'bias', 'fair', 'check', 'ensure', 'validat', 'analy'
    ]
    for pos in positives:
        if pos in text:
            return 1
            
    # Default to 0 if no positive evidence found
    return 0

analysis_df['has_mitigation'] = analysis_df[mitigation_col].apply(classify_mitigation)

# ---------------------------------------------------------
# 3. Statistical Analysis
# ---------------------------------------------------------
summary = analysis_df.groupby('impact_group')['has_mitigation'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total', 'Mitigation_Yes', 'Rate']

print("\nSummary Statistics (Heuristic Parsing):")
print(summary)

# Prepare for Z-test
if 'Rights-Related' in summary.index and 'Safety-Only' in summary.index:
    count_success = np.array([
        summary.loc['Safety-Only', 'Mitigation_Yes'], 
        summary.loc['Rights-Related', 'Mitigation_Yes']
    ])
    nobs = np.array([
        summary.loc['Safety-Only', 'Total'], 
        summary.loc['Rights-Related', 'Total']
    ])
    
    # proportions_ztest
    stat, pval = proportions_ztest(count_success, nobs)
    
    print(f"\nZ-test Results (Safety-Only vs Rights-Related):")
    print(f"Z-statistic: {stat:.4f}")
    print(f"P-value: {pval:.4e}")
    
    # ---------------------------------------------------------
    # 4. Visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    bars = plt.bar(summary.index, summary['Rate'], color=['#FF9999', '#66B2FF'])
    plt.ylabel('Proportion with Disparity Mitigation')
    plt.title('Disparity Mitigation Evidence: Safety-Only vs Rights-Related')
    plt.ylim(0, 1.1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
else:
    print("\nInsufficient data for comparison (one or both groups missing).")
