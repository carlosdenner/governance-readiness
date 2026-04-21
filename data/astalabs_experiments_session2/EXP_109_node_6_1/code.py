import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load dataset
# Reverting to current directory as the previous attempt with '../' failed.
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print("--- Data Loading ---")
print(f"Total EO 13960 records: {len(eo_data)}")

# --- Data Processing ---

# 1. Map Impact Type
# Logic: Group strict 'High'/'Critical' vs 'Moderate'/'Low'. 
# Exclude strictly 'Unknown' or n/a if they don't fit the binary comparison.
def map_impact(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    if 'high' in val_str or 'critical' in val_str or 'rights-impacting' in val_str or 'safety-impacting' in val_str:
        return 'High Impact'
    elif 'moderate' in val_str or 'low' in val_str or 'non-impacting' in val_str:
        return 'Low/Moderate Impact'
    else:
        return None # Exclude Unknown/Unclassified for this specific hypothesis

eo_data['impact_group'] = eo_data['17_impact_type'].apply(map_impact)

# 2. Map Independent Evaluation
def map_eval(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower()
    if 'yes' in val_str and not 'no' in val_str: # Simple 'yes' check, avoiding 'No'
        return 1
    return 0

eo_data['has_indep_eval'] = eo_data['55_independent_eval'].apply(map_eval)

# Drop rows where impact group is undefined
analysis_df = eo_data.dropna(subset=['impact_group'])

print(f"\nRecords after filtering for valid Impact Type: {len(analysis_df)}")
print(analysis_df['impact_group'].value_counts())

# --- Statistical Analysis ---

# Contingency Table
contingency_table = pd.crosstab(analysis_df['impact_group'], analysis_df['has_indep_eval'])
contingency_table.columns = ['No Indep. Eval', 'Has Indep. Eval']
print("\n--- Contingency Table ---")
print(contingency_table)

# Calculate Proportions
props = analysis_df.groupby('impact_group')['has_indep_eval'].agg(['mean', 'count', 'sum'])
props.columns = ['Proportion', 'Total N', 'Count Yes']
print("\n--- Proportions (Audit Rate) ---")
print(props)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n--- Chi-Square Test Results ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpretation
alpha = 0.05
print("\n--- Interpretation ---")
if p < alpha:
    print("Result: Statistically Significant.")
    high_prop = props.loc['High Impact', 'Proportion']
    low_prop = props.loc['Low/Moderate Impact', 'Proportion']
    if high_prop > low_prop:
        print("High Impact systems are significantly MORE likely to have independent evaluations.")
    else:
        print("High Impact systems are significantly LESS likely to have independent evaluations (Counter-intuitive).")
else:
    print("Result: Not Statistically Significant.")
    print("There is no statistical evidence that High Impact systems undergo independent evaluation more often than Low/Moderate systems.")

# Visualization
plt.figure(figsize=(8, 6))
ax = props['Proportion'].plot(kind='bar', color=['skyblue', 'salmon'], alpha=0.8)
plt.title('Independent Evaluation Rate by System Impact')
plt.ylabel('Proportion with Independent Eval')
plt.xlabel('Impact Classification')
plt.ylim(0, max(props['Proportion']) * 1.3) # Add some headroom

# Add value labels
for i, v in enumerate(props['Proportion']):
    ax.text(i, v + 0.005, f"{v:.1%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
