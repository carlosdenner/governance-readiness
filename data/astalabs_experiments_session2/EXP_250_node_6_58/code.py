import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except:
        print("Dataset not found.")
        sys.exit(1)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# 1. Map Autonomy
# Autonomy3 = High
# Autonomy1, Autonomy2 = Low
def map_autonomy(val):
    if pd.isna(val): return np.nan
    if val == 'Autonomy3': return 'High'
    if val in ['Autonomy1', 'Autonomy2']: return 'Low'
    return np.nan

aiid_df['autonomy_group'] = aiid_df['Autonomy Level'].apply(map_autonomy)

# 2. Map Harm
# Physical: 'tangible harm definitively occurred'
# Intangible: 'yes' in Special Interest Intangible Harm AND NOT Physical
def map_harm(row):
    tangible = str(row['Tangible Harm'])
    intangible = str(row['Special Interest Intangible Harm'])
    
    is_physical = 'tangible harm definitively occurred' in tangible
    is_intangible = 'yes' == intangible.lower()
    
    if is_physical:
        return 'Physical'
    elif is_intangible:
        return 'Intangible'
    else:
        return np.nan

aiid_df['harm_group'] = aiid_df.apply(map_harm, axis=1)

# 3. Filter for valid rows
analysis_df = aiid_df.dropna(subset=['autonomy_group', 'harm_group'])

print(f"Valid incidents for analysis: {len(analysis_df)}")
print("Distribution:")
print(analysis_df.groupby(['autonomy_group', 'harm_group']).size())

if len(analysis_df) < 5:
    print("Insufficient data for statistical testing.")
    sys.exit(0)

# 4. Chi-Square Test
contingency = pd.crosstab(analysis_df['autonomy_group'], analysis_df['harm_group'])
print("\nContingency Table:")
print(contingency)

chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")

# 5. Plotting
# Calculate proportion of Physical harm
props = contingency.div(contingency.sum(axis=1), axis=0)
physical_props = props['Physical'] if 'Physical' in props.columns else pd.Series([0,0], index=['High', 'Low'])

plt.figure(figsize=(8, 6))
bars = plt.bar(physical_props.index, physical_props.values, color=['#d62728', '#1f77b4'])
plt.ylabel('Proportion of Incidents Involving Physical Harm')
plt.title('Physical Harm Rates by Autonomy Level')
plt.ylim(0, 1.0)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{height:.1%}', ha='center', va='bottom')

plt.show()