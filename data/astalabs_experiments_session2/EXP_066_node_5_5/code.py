import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# [debug]
print("Starting Governance Bundling experiment...")

# 1. Load Data
file_name = 'astalabs_discovery_all_data.csv'
if os.path.exists(file_name):
    file_path = file_name
elif os.path.exists(f'../{file_name}'):
    file_path = f'../{file_name}'
else:
    print("Error: Dataset not found.")
    sys.exit(1)

try:
    df = pd.read_csv(file_path, low_memory=False)
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Loaded EO 13960 subset: {len(df_eo)} rows")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# 2. Define Semantic Parsers
def parse_bias_mitigation(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    
    # Exclusion criteria (explicit statements that mitigation is N/A or not performed)
    # "The AI does not take into account..." implies risk avoidance, not active mitigation control.
    exclusions = [
        'n/a', 'none', 'not applicable', 'no demographic', 
        'not safety', 'waived', 'does not take into account', 
        'no testing', 'not leveraged'
    ]
    if any(ex in val_str for ex in exclusions):
        return 0
    
    # Inclusion criteria (Active governance verbs)
    inclusions = [
        'test', 'eval', 'monitor', 'review', 'assess', 
        'human', 'guardrail', 'check', 'audit', 'mitigat', 
        'ensure', 'verify', 'feedback'
    ]
    if any(inc in val_str for inc in inclusions):
        return 1
        
    return 0

def parse_real_world_testing(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    
    # Explicit No
    if 'no testing' in val_str:
        return 0
    
    # Explicit Yes or strong indicators of *Operational* testing
    # Note: "Benchmark evaluation" often explicitly says "has not been tested in an operational environment"
    if 'operational environment' in val_str or 'yes' == val_str or val_str.startswith('yes,'):
        return 1
        
    return 0

# 3. Apply Parsing
df_eo['has_bias_mitigation'] = df_eo['62_disparity_mitigation'].apply(parse_bias_mitigation)
df_eo['has_rw_testing'] = df_eo['53_real_world_testing'].apply(parse_real_world_testing)

# 4. Analysis
contingency_table = pd.crosstab(
    df_eo['has_bias_mitigation'], 
    df_eo['has_rw_testing'], 
    rownames=['Bias Mitigation'], 
    colnames=['Real-World Testing']
)

print("\n--- Contingency Table ---")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Phi Coefficient
n = contingency_table.sum().sum()
phi = np.sqrt(chi2 / n)

print(f"\n--- Statistical Results ---")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Phi Coefficient: {phi:.4f}")

# Interpretation
interpretation = ""
if p < 0.05:
    interpretation += "Significant correlation found. "
    if phi > 0.5:
        interpretation += "Strong association."
    elif phi > 0.3:
        interpretation += "Moderate association."
    elif phi > 0.1:
        interpretation += "Weak association."
    else:
        interpretation += "Negligible association."
else:
    interpretation += "No significant correlation found (Independence cannot be rejected)."

print(f"Interpretation: {interpretation}")

# Conditional Probabilities
# P(Testing | Bias) = TP / (TP + FN)
# P(Testing | No Bias) = FP / (FP + TN)
# Using loc to access safely
try:
    tp = contingency_table.loc[1, 1] if 1 in contingency_table.index and 1 in contingency_table.columns else 0
    fn = contingency_table.loc[1, 0] if 1 in contingency_table.index and 0 in contingency_table.columns else 0
    fp = contingency_table.loc[0, 1] if 0 in contingency_table.index and 1 in contingency_table.columns else 0
    tn = contingency_table.loc[0, 0] if 0 in contingency_table.index and 0 in contingency_table.columns else 0
    
    p_test_given_bias = tp / (tp + fn) if (tp + fn) > 0 else 0
    p_test_given_no_bias = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\nLikelihood of Testing if Bias Mitigation exists: {p_test_given_bias:.2%}")
    print(f"Likelihood of Testing if Bias Mitigation is ABSENT: {p_test_given_no_bias:.2%}")
except Exception as e:
    print(f"Could not calculate conditional probabilities: {e}")

# 5. Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Governance Bundling:\nBias Mitigation vs. Real-World Testing')
plt.xlabel('Real-World Testing Implemented')
plt.ylabel('Bias Mitigation Implemented')
plt.xticks([0.5, 1.5], ['No', 'Yes'])
plt.yticks([0.5, 1.5], ['No', 'Yes'])
plt.tight_layout()
plt.show()
