import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys

print("Starting 'Paper Compliance' Hypothesis Experiment (Attempt 2)...")

# 1. Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# 2. Filter for 'eo13960_scored'
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset shape: {subset.shape}")

# 3. Define Cleaning Logic

def clean_impact_assessment(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ['yes', 'true', '1']:
        return 'Yes'
    elif s in ['no', 'false', '0', 'planned or in-progress.']:
        return 'No'
    return np.nan

def clean_real_world_testing(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    
    # Check for specific keywords based on dataset inspection
    if s == 'yes':
        return 'Yes'
    if 'operational environment' in s and 'not been tested' not in s:
        # Covers "Performance evaluation..." and "Impact evaluation..."
        return 'Yes'
    if 'no testing' in s or 'waived' in s or 'benchmark evaluation' in s:
        # Benchmark explicitly says "not been tested in an operational environment"
        return 'No'
    if 'no' == s:
        return 'No'
    
    return np.nan

# 4. Apply Cleaning
subset['Assessment_Clean'] = subset['52_impact_assessment'].apply(clean_impact_assessment)
subset['Testing_Clean'] = subset['53_real_world_testing'].apply(clean_real_world_testing)

# 5. Drop NaNs
analysis_df = subset.dropna(subset=['Assessment_Clean', 'Testing_Clean'])
print(f"Rows with valid data: {len(analysis_df)}")

# 6. Generate Contingency Table
contingency = pd.crosstab(analysis_df['Assessment_Clean'], analysis_df['Testing_Clean'])
print("\nContingency Table (Count):")
print(contingency)

# Row percentages
row_pct = pd.crosstab(analysis_df['Assessment_Clean'], analysis_df['Testing_Clean'], normalize='index') * 100
print("\nContingency Table (Row %):")
print(row_pct.round(2))

# 7. Statistical Test
if contingency.size == 0:
    print("Empty contingency table.")
    sys.exit(0)

chi2, p, dof, expected = stats.chi2_contingency(contingency)
print("\n--- Chi-Square Test Results ---")
print(f"Chi2: {chi2:.4f}, p-value: {p:.4e}, dof: {dof}")

# Fisher's Exact Test if 2x2
if contingency.shape == (2, 2):
    oddsratio, p_fisher = stats.fisher_exact(contingency)
    print(f"Fisher's Exact Test p-value: {p_fisher:.4e}")
    print(f"Odds Ratio: {oddsratio:.4f}")
    final_p = p_fisher
else:
    final_p = p

if final_p < 0.05:
    print("Result: Significant association found.")
else:
    print("Result: No significant association found.")

# 8. Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
plt.title('Assessment vs Testing (Count)')

plt.subplot(1, 2, 2)
sns.heatmap(row_pct, annot=True, fmt='.1f', cmap='Greens')
plt.title('Assessment vs Testing (Row %)')
plt.tight_layout()
plt.show()
