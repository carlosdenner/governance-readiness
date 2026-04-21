import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import sys

# Load dataset
# Reverting to current directory as previous attempt with parent directory failed
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    sys.exit(1)

# Filter for EO 13960 scored data
subset = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Step 1: Define Impact Groups ---
# Column: 17_impact_type

def categorize_impact(val):
    if not isinstance(val, str):
        return None
    val_lower = val.lower()
    has_rights = 'rights' in val_lower
    has_safety = 'safety' in val_lower
    
    if has_rights and not has_safety:
        return 'Rights-Impacting'
    elif has_safety and not has_rights:
        return 'Safety-Impacting'
    # We exclude 'Both' to get a cleaner contrast, or we could keep it. 
    # The hypothesis contrasts Rights vs Safety specifically.
    return None

subset['impact_group'] = subset['17_impact_type'].apply(categorize_impact)

# Filter for distinct groups
analysis_df = subset[subset['impact_group'].isin(['Rights-Impacting', 'Safety-Impacting'])].copy()

print(f"Total EO13960 records: {len(subset)}")
print(f"Analysis Base (Rights vs Safety exclusive): {len(analysis_df)} systems")
print(analysis_df['impact_group'].value_counts())
print("-" * 30)

# --- Step 2: Semantic Parsing for Controls ---

# Helper to parse binary controls
def parse_control(text, positive_keywords, negative_keywords):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    # Check negative first
    for kw in negative_keywords:
        if kw in text_lower:
            return 0
    # Check positive
    for kw in positive_keywords:
        if kw in text_lower:
            return 1
    return 0

# Keywords for Bias Mitigation (62_disparity_mitigation)
bias_pos = ['assess', 'analy', 'test', 'eval', 'monitor', 'review', 'audit', 'check', 'ensur', 'mitigat', 'perform']
bias_neg = ['no ', 'none', 'n/a', 'not applicable', 'unknown', 'tbd', 'waived']

# Keywords for Real-World Testing (53_real_world_testing)
test_pos = ['yes', 'pilot', 'beta', 'field', 'real', 'deploy', 'environment', 'test', 'eval', 'operat']
test_neg = ['no ', 'none', 'n/a', 'lab', 'bench', 'not applicable']

analysis_df['has_bias_mitigation'] = analysis_df['62_disparity_mitigation'].apply(
    lambda x: parse_control(x, bias_pos, bias_neg)
)

analysis_df['has_rw_testing'] = analysis_df['53_real_world_testing'].apply(
    lambda x: parse_control(x, test_pos, test_neg)
)

# --- Step 3: Statistical Analysis ---

def run_chi2(df, group_col, target_col, label):
    ct = pd.crosstab(df[group_col], df[target_col])
    
    # Check if we have enough data
    if ct.empty:
        print(f"Not enough data for {label}")
        return {}

    chi2, p, dof, ex = chi2_contingency(ct)
    
    # Calculate rates (mean of 0s and 1s gives the proportion of 1s)
    rates = df.groupby(group_col)[target_col].mean()
    
    print(f"\n--- Analysis: {label} ---")
    print("Contingency Table (Count):")
    print(ct)
    print("\nCompliance Rates:")
    print(rates)
    print(f"\nChi2: {chi2:.2f}, p-value: {p:.4f}")
    return rates

print("\nHYPOTHESIS TEST 1: Rights-Impacting systems are more likely to have Bias Mitigation.")
rates_bias = run_chi2(analysis_df, 'impact_group', 'has_bias_mitigation', 'Bias Mitigation Compliance')

print("\nHYPOTHESIS TEST 2: Safety-Impacting systems are more likely to have Real-World Testing.")
rates_test = run_chi2(analysis_df, 'impact_group', 'has_rw_testing', 'Real-World Testing Compliance')

# --- Step 4: Visualization ---
if not rates_bias.empty and not rates_test.empty:
    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = ['Bias Mitigation', 'Real-World Testing']
    
    # Extract rates safely
    r_bias_rights = rates_bias.get('Rights-Impacting', 0)
    r_bias_safety = rates_bias.get('Safety-Impacting', 0)
    
    r_test_rights = rates_test.get('Rights-Impacting', 0)
    r_test_safety = rates_test.get('Safety-Impacting', 0)

    rights_scores = [r_bias_rights, r_test_rights]
    safety_scores = [r_bias_safety, r_test_safety]

    x = np.arange(len(x_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, rights_scores, width, label='Rights-Impacting', color='skyblue')
    rects2 = ax.bar(x + width/2, safety_scores, width, label='Safety-Impacting', color='orange')

    ax.set_ylabel('Compliance Rate (0-1)')
    ax.set_title('Governance Priorities: Rights vs Safety')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    plt.ylim(0, 1.1)
    plt.show()
