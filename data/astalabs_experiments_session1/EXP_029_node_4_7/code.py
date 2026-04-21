import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import os

# Handle file loading robustly
filename = 'step3_incident_coding.csv'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    print(f"File {filename} not found.")
    exit(1)

df = pd.read_csv(filepath)
print(f"Loaded dataset with {len(df)} records.")

# 1. Clean and Inspect Columns
print("\n--- unique values in trust_integration_split ---")
print(df['trust_integration_split'].unique())

# Normalize column for filtering
df['split_norm'] = df['trust_integration_split'].astype(str).str.strip().str.lower()

# 2. Filter out 'both' to isolate dominant factors
# We look for rows that do NOT contain 'both' but ARE valid
# Based on metadata, we expect roughly 6 records here
subset = df[~df['split_norm'].str.contains('both')].copy()

# 3. Categorize into Trust vs Integration
def classify_split(val):
    if 'trust' in val:
        return 'Trust'
    elif 'integration' in val:
        return 'Integration'
    return None

subset['group'] = subset['split_norm'].apply(classify_split)
subset = subset.dropna(subset=['group'])

print(f"\nFiltered subset size: {len(subset)}")
print(subset['group'].value_counts())

# 4. Categorize Failure Mode
def classify_failure(val):
    val_str = str(val).lower()
    if 'prevention' in val_str:
        return 'Prevention'
    else:
        return 'Detection/Response'

subset['failure_type'] = subset['failure_mode'].apply(classify_failure)

# 5. Contingency Table
ct = pd.crosstab(subset['group'], subset['failure_type'])
print("\n--- Contingency Table ---")
print(ct)

# Ensure 2x2 shape for Fisher Test
expected_rows = ['Integration', 'Trust']
expected_cols = ['Detection/Response', 'Prevention']
ct_full = ct.reindex(index=expected_rows, columns=expected_cols, fill_value=0)
print("\n--- Full Table for Stats ---")
print(ct_full)

# 6. Statistical Test (Fisher's Exact)
if ct_full.values.sum() > 0:
    odds_ratio, p_value = fisher_exact(ct_full)
    print(f"\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Conclusion: Statistically significant association found.")
    else:
        print("Conclusion: No statistically significant association (likely due to small sample size).")
else:
    print("Not enough data for statistical testing.")

# 7. Visualization
if not ct_full.empty and ct_full.values.sum() > 0:
    plt.figure(figsize=(8, 6))
    ct_full.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], ax=plt.gca())
    plt.title('Failure Mode by Readiness Dominance')
    plt.xlabel('Dominant Readiness Gap')
    plt.ylabel('Incident Count')
    plt.xticks(rotation=0)
    plt.legend(title='Failure Mode')
    plt.tight_layout()
    plt.show()
