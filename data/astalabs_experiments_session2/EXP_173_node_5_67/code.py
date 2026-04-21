import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import sys
import os

# 1. Load Data
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, '../' + filename]
filepath = None
for p in possible_paths:
    if os.path.exists(p):
        filepath = p
        break

if not filepath:
    print("Dataset not found.")
    sys.exit(1)

print(f"Loading dataset from {filepath}...")
df = pd.read_csv(filepath, low_memory=False)

# 2. Filter for AIID Incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID Incidents: {len(aiid)}")

# 3. Map Failure Category
def map_failure(val):
    if pd.isna(val):
        return None
    text = str(val).lower()
    
    # Bias / Fairness Keywords
    if any(k in text for k in ['bias', 'fair', 'discriminat', 'equit', 'civil rights', 'demographic']):
        return 'Bias/Fairness'
    
    # Safety / Robustness Keywords
    if any(k in text for k in ['mistake', 'error', 'crash', 'safety', 'robust', 'generalization', 'context', 'hazard', 'unsafe', 'performance', 'failure']):
        return 'Safety/Robustness'
    
    return None

aiid['Failure_Category'] = aiid['Known AI Technical Failure'].apply(map_failure)

# 4. Map Harm Category
# Logic: Tangible = 'definitively occurred', Intangible = 'yes' in Special Interest
def map_harm(row):
    tangible_val = str(row.get('Tangible Harm', '')).lower()
    is_tangible = 'definitively occurred' in tangible_val
    
    intangible_val = str(row.get('Special Interest Intangible Harm', '')).lower()
    is_intangible = 'yes' in intangible_val
    
    # Classify
    if is_tangible and not is_intangible:
        return 'Tangible'
    elif is_intangible and not is_tangible:
        return 'Intangible'
    elif is_tangible and is_intangible:
        return 'Both' # Excluded from Chi-square to ensure mutual exclusivity
    else:
        return None

aiid['Harm_Category'] = aiid.apply(map_harm, axis=1)

# 5. Filter for Analysis
# We exclude 'Both' to strictly test the divergence between Tangible and Intangible outcomes
analysis_df = aiid.dropna(subset=['Failure_Category', 'Harm_Category'])
analysis_df = analysis_df[analysis_df['Harm_Category'] != 'Both']

print(f"Rows for analysis (Exclusive categories): {len(analysis_df)}")
print("\nCounts by Failure Category:")
print(analysis_df['Failure_Category'].value_counts())
print("\nCounts by Harm Category:")
print(analysis_df['Harm_Category'].value_counts())

# 6. Statistical Test
if len(analysis_df) < 5:
    print("\nInsufficient data for Chi-square test.")
else:
    contingency = pd.crosstab(analysis_df['Failure_Category'], analysis_df['Harm_Category'])
    print("\nContingency Table:")
    print(contingency)
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Significant association found (Reject Null Hypothesis).")
    else:
        print("Result: No significant association found (Fail to reject Null Hypothesis).")
    
    # 7. Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
    plt.title('Technical Failure Mode vs Harm Domain')
    plt.ylabel('Technical Failure')
    plt.xlabel('Harm Domain (Exclusive)')
    plt.tight_layout()
    plt.show()
