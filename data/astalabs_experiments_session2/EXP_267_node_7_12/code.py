import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Fallback if file is in parent directory (standard for some envs)
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# Clean and Prepare Data
# We only analyze rows where both Autonomy and Harm are coded
analysis_df = aiid_df.dropna(subset=['Autonomy Level', 'Tangible Harm'])

# Exclude 'unclear' values
analysis_df = analysis_df[
    (analysis_df['Autonomy Level'] != 'unclear') & 
    (analysis_df['Tangible Harm'] != 'unclear')
]

# Mapping Functions
def map_autonomy(val):
    # Autonomy3 is typically 'System is autonomous' (High)
    if val == 'Autonomy3':
        return 'High'
    # Autonomy1 ('System is human-in-the-loop') and Autonomy2 ('System is human-supervised') (Low)
    elif val in ['Autonomy1', 'Autonomy2']:
        return 'Low'
    return None

def map_harm(val):
    # Hypothesis focuses on 'Tangible Harm' (Physical injury/death)
    # We define 'Tangible' as cases where it definitively occurred.
    if val == 'tangible harm definitively occurred':
        return 'Tangible (Physical)'
    else:
        # Includes near-misses, issues, no tangible harm
        return 'Intangible/None'

# Apply mappings
analysis_df['Autonomy_Group'] = analysis_df['Autonomy Level'].apply(map_autonomy)
analysis_df['Harm_Group'] = analysis_df['Tangible Harm'].apply(map_harm)

# Drop any rows that failed mapping (though previous filter should catch them)
analysis_df = analysis_df.dropna(subset=['Autonomy_Group', 'Harm_Group'])

# Generate Summary Stats
print(f"Total Incidents Analyzed: {len(analysis_df)}")
print(analysis_df['Autonomy_Group'].value_counts())

# Create Contingency Table
contingency = pd.crosstab(analysis_df['Autonomy_Group'], analysis_df['Harm_Group'])
print("\n--- Contingency Table ---")
print(contingency)

# Calculate Rates
high_auto_total = contingency.loc['High'].sum() if 'High' in contingency.index else 0
high_auto_tangible = contingency.loc['High', 'Tangible (Physical)'] if 'High' in contingency.index and 'Tangible (Physical)' in contingency.columns else 0
rate_high = high_auto_tangible / high_auto_total if high_auto_total > 0 else 0

low_auto_total = contingency.loc['Low'].sum() if 'Low' in contingency.index else 0
low_auto_tangible = contingency.loc['Low', 'Tangible (Physical)'] if 'Low' in contingency.index and 'Tangible (Physical)' in contingency.columns else 0
rate_low = low_auto_tangible / low_auto_total if low_auto_total > 0 else 0

print(f"\nHigh Autonomy Tangible Harm Rate: {rate_high:.1%} ({high_auto_tangible}/{high_auto_total})")
print(f"Low Autonomy Tangible Harm Rate:  {rate_low:.1%} ({low_auto_tangible}/{low_auto_total})")

# Statistical Test
# Fisher's Exact Test is appropriate given the relatively small sample sizes in some cells
if contingency.size == 4:
    odds_ratio, p_value = stats.fisher_exact(contingency.loc[['High', 'Low'], ['Tangible (Physical)', 'Intangible/None']])
    print(f"\nFisher's Exact Test P-value: {p_value:.5f}")
    print(f"Odds Ratio: {odds_ratio:.2f}")
    
    if p_value < 0.05:
        print("Result: Significant correlation found.")
    else:
        print("Result: No significant correlation found.")
else:
    print("\nInsufficient data structure for 2x2 statistical test.")

# Visualization
plt.figure(figsize=(8, 5))
plt.bar(['High Autonomy', 'Low Autonomy'], [rate_high, rate_low], color=['#d62728', '#1f77b4'])
plt.ylabel('Rate of Tangible Harm')
plt.title('Tangible Harm Rates by Autonomy Level')
plt.ylim(0, 1.0)
plt.show()
