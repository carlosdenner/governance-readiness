import pandas as pd
import scipy.stats as stats
import sys

# Load the dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback for alternative directory structures if needed, though instruction says same level
    # Trying relative path based on previous prompt hint
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

print(f"Total AIID Incidents: {len(aiid_df)}")

# --- Corrected Preprocessing ---

# 1. Clean Autonomy Level
# Mapping based on typical definitions for Autonomy1/2/3 in this dataset context:
# Autonomy1 = System recommends action, human decides (Low)
# Autonomy2 = System initiates action, human can override (High/Medium)
# Autonomy3 = System initiates action, human out of loop (High)

def map_autonomy_corrected(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    
    if val_str == 'Autonomy1':
        return 'Low'
    elif val_str in ['Autonomy2', 'Autonomy3']:
        return 'High'
    return None

# 2. Clean Intentional Harm
# Values start with 'Yes' or 'No'

def map_intent_corrected(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip().lower()
    
    if val_str.startswith('yes'):
        return 'Intentional'
    elif val_str.startswith('no'):
        return 'Unintentional'
    return None

aiid_df['autonomy_bin'] = aiid_df['Autonomy Level'].apply(map_autonomy_corrected)
aiid_df['intent_bin'] = aiid_df['Intentional Harm'].apply(map_intent_corrected)

# Drop records where we couldn't classify
analysis_df = aiid_df.dropna(subset=['autonomy_bin', 'intent_bin'])

print(f"Records available for analysis: {len(analysis_df)}")

# --- Analysis ---

# Create Contingency Table
contingency_table = pd.crosstab(analysis_df['autonomy_bin'], analysis_df['intent_bin'])
print("\nContingency Table (Autonomy vs Intentionality):")
print(contingency_table)

# Check if we have data for all cells to be safe
if contingency_table.empty:
    print("\nNo valid data found for contingency table.")
else:
    # Extract values for Fisher's Exact Test
    # Structure: [[Low_Intentional, Low_Unintentional], [High_Intentional, High_Unintentional]]
    
    # Helper to safely get value
    def get_val(r, c):
        try:
            return contingency_table.loc[r, c]
        except KeyError:
            return 0

    low_intent = get_val('Low', 'Intentional')
    low_unintent = get_val('Low', 'Unintentional')
    high_intent = get_val('High', 'Intentional')
    high_unintent = get_val('High', 'Unintentional')
    
    table_for_stats = [[low_intent, low_unintent], [high_intent, high_unintent]]
    
    print(f"\nMatrix for Fisher's Test:\n {table_for_stats}")
    
    if (low_intent + low_unintent == 0) or (high_intent + high_unintent == 0):
         print("Cannot run test: One row is empty.")
    else:
        # We expect Low Autonomy to be MORE associated with Intentional Harm
        # So Odds Ratio > 1
        odds_ratio, p_value = stats.fisher_exact(table_for_stats, alternative='greater')
        
        print(f"\n--- Statistical Results ---")
        print(f"Odds Ratio: {odds_ratio:.4f}")
        print(f"P-Value (one-sided 'greater'): {p_value:.4f}")
        
        # Calculate percentages for clearer interpretation
        pct_low_intent = (low_intent / (low_intent + low_unintent)) * 100
        pct_high_intent = (high_intent / (high_intent + high_unintent)) * 100
        
        print(f"\nInterpretation:")
        print(f"% of Low Autonomy incidents that were Intentional: {pct_low_intent:.1f}%")
        print(f"% of High Autonomy incidents that were Intentional: {pct_high_intent:.1f}%")
        
        if p_value < 0.05:
            print("\nResult: Statistically Significant. The hypothesis is supported.")
        else:
            print("\nResult: Not Statistically Significant.")
