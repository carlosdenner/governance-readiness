import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print("--- Data Loading ---")
print(f"Total EO 13960 records: {len(eo_data)}")

# Inspect columns for mapping
print("\nUnique values in '16_dev_stage':")
print(eo_data['16_dev_stage'].unique())
print("\nUnique values in '29_contains_pii':")
print(eo_data['29_contains_pii'].unique())

# 1. Map Development Stage
# Hypothesized mapping based on federal IT standards:
# Operational: 'Operation and maintenance', 'Implemented'
# Development: 'Development and acquisition', 'Planned', 'Research and development'

def map_stage(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    if 'operation' in val_lower or 'implemented' in val_lower or 'use' in val_lower:
        return 'Operational'
    elif 'development' in val_lower or 'planned' in val_lower or 'acquisition' in val_lower:
        return 'Development'
    else:
        return None # Exclude other categories (e.g. retired) if ambiguous

eo_data['stage_group'] = eo_data['16_dev_stage'].apply(map_stage)

# 2. Map PII
def map_pii(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    if 'yes' in val_lower:
        return True
    elif 'no' in val_lower:
        return False
    return None

eo_data['has_pii'] = eo_data['29_contains_pii'].apply(map_pii)

# Filter for valid data
valid_data = eo_data.dropna(subset=['stage_group', 'has_pii'])

print(f"\nRecords after filtering for valid Stage and PII info: {len(valid_data)}")

# 3. Analysis
# Contingency Table
contingency_table = pd.crosstab(valid_data['stage_group'], valid_data['has_pii'])
print("\nContingency Table (Stage vs Has PII):")
print(contingency_table)

# Calculate Proportions
results = valid_data.groupby('stage_group')['has_pii'].agg(['count', 'sum', 'mean'])
results.columns = ['Total', 'With_PII', 'Proportion']
print("\nProportions by Stage:")
print(results)

# 4. Statistical Test (Chi-Square Test of Independence)
# We use Chi-Square as it's equivalent to Z-test for proportions with 2 groups but handles the contingency table directly
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n--- Statistical Test Results (Chi-Square) ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("Result: Significant difference found. Hypothesis supported (if direction matches) or rejected (if opposite).")
else:
    print("Result: No significant difference found.")

# Check directionality
if 'Operational' in results.index and 'Development' in results.index:
    op_prop = results.loc['Operational', 'Proportion']
    dev_prop = results.loc['Development', 'Proportion']
    print(f"Operational PII Rate: {op_prop:.2%}")
    print(f"Development PII Rate: {dev_prop:.2%}")
    if op_prop > dev_prop:
        print("Direction: Operational systems have a HIGHER rate of PII usage.")
    else:
        print("Direction: Operational systems have a LOWER or EQUAL rate of PII usage.")

# 5. Visualization
plt.figure(figsize=(8, 6))
sns.barplot(x=results.index, y='Proportion', data=results.reset_index(), palette='viridis')
plt.ylabel('Proportion of Systems Containing PII')
plt.title('PII Usage by Lifecycle Stage')
plt.ylim(0, 1.0)
for index, row in results.reset_index().iterrows():
    plt.text(index, row['Proportion'] + 0.02, f"{row['Proportion']:.1%}", ha='center')
plt.show()
