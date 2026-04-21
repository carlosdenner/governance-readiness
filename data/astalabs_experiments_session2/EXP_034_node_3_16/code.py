import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

print("Dataset loaded. Processing 'eo13960_scored'...")

eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# Columns
col_pii = '29_contains_pii'
col_bias = '62_disparity_mitigation'

# --- Step 1: Clean PII Variable ---
# We only keep rows where PII status is explicitly 'Yes' or 'No' to ensure validity.
def clean_pii(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ['yes', 'y', 'true', '1']:
        return 1
    elif s in ['no', 'n', 'false', '0']:
        return 0
    return np.nan

eo_df['pii_binary'] = eo_df[col_pii].apply(clean_pii)

# Drop rows where PII is unknown (NaN)
analysis_df = eo_df.dropna(subset=['pii_binary']).copy()
print(f"Rows after filtering for valid PII (Yes/No): {len(analysis_df)}")
print(f"PII Distribution:\n{analysis_df['pii_binary'].value_counts()}")

# --- Step 2: Clean Bias Mitigation Variable ---
# Logic: NaN is treated as 0 (No mitigation listed).
# Text is analyzed: 'N/A', 'None', 'Not applicable' -> 0.
# Substantive text -> 1.

def clean_bias_mitigation(val):
    if pd.isna(val):
        return 0  # Treat missing as no mitigation
    
    text = str(val).strip().lower()
    
    # Check for empty or essentially empty strings
    if not text or text == 'nan':
        return 0
        
    # Keywords indicating LACK of mitigation or irrelevance
    negatives = [
        r'^n/a',
        r'^none',
        r'not applicable',
        r'no demographic',
        r'does not use',
        r'not safety',
        r'waived',
        r'not using pii'
    ]
    
    for pattern in negatives:
        if re.search(pattern, text):
            return 0
            
    # If text is present and not a negative keyword, assume it describes a mitigation
    return 1

analysis_df['bias_binary'] = analysis_df[col_bias].apply(clean_bias_mitigation)

print(f"Bias Mitigation Distribution:\n{analysis_df['bias_binary'].value_counts()}")

# --- Step 3: Analysis ---
if len(analysis_df) == 0:
    print("Error: No data remaining.")
else:
    # Contingency Table
    contingency = pd.crosstab(analysis_df['pii_binary'], analysis_df['bias_binary'])
    
    # Ensure 2x2 shape
    for i in [0, 1]:
        if i not in contingency.index:
            contingency.loc[i] = [0, 0]
        if i not in contingency.columns:
            contingency[i] = 0
            
    contingency = contingency.sort_index().sort_index(axis=1)
    contingency.index = ['No PII', 'Has PII']
    contingency.columns = ['No Mitigation', 'Has Mitigation']
    
    print("\n--- Contingency Table ---")
    print(contingency)
    
    # Chi-square
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square: {chi2:.4f}, p-value: {p:.4e}")
    
    # Probabilities
    # P(Mitigation | PII)
    p_mit_pii = contingency.loc['Has PII', 'Has Mitigation'] / contingency.loc['Has PII'].sum()
    # P(Mitigation | No PII)
    p_mit_no_pii = contingency.loc['No PII', 'Has Mitigation'] / contingency.loc['No PII'].sum()
    
    print(f"P(Mitigation | PII)    = {p_mit_pii:.2%}")
    print(f"P(Mitigation | No PII) = {p_mit_no_pii:.2%}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Greens')
    plt.title('PII Presence vs. Fairness/Bias Mitigation')
    plt.show()
    
    # Conclusion
    print("\n--- Conclusion ---")
    if p < 0.05:
        print("Statistically Significant Association detected.")
    else:
        print("No Statistically Significant Association (Supports 'Disconnect' Hypothesis).")
