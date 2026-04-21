import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

# Load the dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
df_incidents = df[df['source_table'] == 'aiid_incidents'].copy()

print("--- Data Loading ---")
print(f"Total Incidents Loaded: {len(df_incidents)}")

# --- Data Cleaning & Mapping ---

# Map Autonomy Level based on debug findings
# Autonomy3 -> High (Autonomous)
# Autonomy1, Autonomy2 -> Low (Assisted/Human-in-the-loop)
def map_autonomy(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).strip()
    if val_str == 'Autonomy3':
        return 'High'
    elif val_str in ['Autonomy1', 'Autonomy2']:
        return 'Low'
    return 'Unknown'

# Map Tangible Harm based on debug findings
# 'tangible harm definitively occurred' -> Tangible
# All other valid categories (near-misses, issues, no harm) -> Intangible (i.e., did not result in tangible harm)
def map_harm(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).strip()
    
    if val_str == 'tangible harm definitively occurred':
        return 'Tangible'
    elif val_str in ['no tangible harm, near-miss, or issue', 
                     'non-imminent risk of tangible harm (an issue) occurred',
                     'imminent risk of tangible harm (near miss) did occur']:
        return 'Intangible'
    return 'Unknown'

# Apply mappings
df_incidents['Autonomy_Group'] = df_incidents['Autonomy Level'].apply(map_autonomy)
df_incidents['Harm_Type'] = df_incidents['Tangible Harm'].apply(map_harm)

# Filter out Unknowns for the analysis
analysis_df = df_incidents[
    (df_incidents['Autonomy_Group'] != 'Unknown') & 
    (df_incidents['Harm_Type'] != 'Unknown')
].copy()

print("\n--- Mapped Data Distribution (Analysis Subset) ---")
print(pd.crosstab(analysis_df['Autonomy_Group'], analysis_df['Harm_Type']))

# --- Statistical Test ---
# Contingency Table structure: 
#           Tangible | Intangible
# High      a          b
# Low       c          d

contingency_table = pd.crosstab(
    analysis_df['Autonomy_Group'], 
    analysis_df['Harm_Type']
).reindex(index=['High', 'Low'], columns=['Tangible', 'Intangible'], fill_value=0)

print("\n--- Contingency Table for Statistical Test ---")
print(contingency_table)

if contingency_table.values.sum() == 0:
    print("\nInsufficient data for statistical test.")
else:
    # Fisher's Exact Test
    # Alternative='greater' tests if High Autonomy is MORE associated with Tangible Harm than Low Autonomy is.
    odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')

    print("\n--- Fisher's Exact Test Results ---")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Calculate percentages for context
    high_sum = contingency_table.loc['High'].sum()
    low_sum = contingency_table.loc['Low'].sum()
    
    high_tangible_rate = (contingency_table.loc['High', 'Tangible'] / high_sum) if high_sum > 0 else 0
    low_tangible_rate = (contingency_table.loc['Low', 'Tangible'] / low_sum) if low_sum > 0 else 0

    print(f"\nHigh Autonomy Tangible Rate: {high_tangible_rate:.2%} ({contingency_table.loc['High', 'Tangible']}/{high_sum})")
    print(f"Low Autonomy Tangible Rate:  {low_tangible_rate:.2%} ({contingency_table.loc['Low', 'Tangible']}/{low_sum})")

    if p_value < 0.05:
        print("\nResult: Statistically Significant. The Autonomy-Harm Hypothesis is supported.")
    else:
        print("\nResult: Not Statistically Significant. The Autonomy-Harm Hypothesis is NOT supported.")
