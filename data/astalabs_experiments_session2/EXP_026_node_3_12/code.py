import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

# Load the dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    print("Dataset not found at astalabs_discovery_all_data.csv")
    sys.exit(1)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID Incidents Loaded: {len(aiid)}")

# --- Define Mapping Functions based on previous debug output ---

def map_autonomy(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    
    # Based on observed values: Autonomy1, Autonomy2, Autonomy3
    # Mapping Assumption: Autonomy1 = Low/Assisted, Autonomy3 = High/Autonomous
    if s in ['Autonomy3', 'Autonomy4', 'Autonomy5', 'High', 'Full Autonomy']:
        return 'High Autonomy'
    if s in ['Autonomy0', 'Autonomy1', 'Low', 'Assisted', 'No Autonomy']:
        return 'Low Autonomy'
        
    # Fallback for string matching if exact match fails
    s_lower = s.lower()
    if 'autonomy3' in s_lower or 'high' in s_lower:
        return 'High Autonomy'
    if 'autonomy1' in s_lower or 'low' in s_lower:
        return 'Low Autonomy'
        
    return None

def map_failure(val):
    if pd.isna(val):
        return None
    s = str(val).lower().strip()
    
    # Robustness: Failures of capability/reliability/security
    if any(x in s for x in ['robustness', 'generalization', 'context', 'gaming', 'attack', 'adversarial', 'distributional', 'reliability', 'hardware']):
        return 'Robustness'
        
    # Specification: Alignment issues, wrong objective, harmful generation (hallucination/misinfo often fit here in CSET taxonomy)
    if any(x in s for x in ['specification', 'objective', 'goal', 'alignment', 'misinformation', 'harmful', 'unsafe', 'bias']):
        # Note: Bias is tricky, but often categorized as spec/alignment in broader governance contexts if not explicitly 'fairness'
        return 'Specification'
        
    # Human-Interaction: Operator error, transparency, use error
    if any(x in s for x in ['human', 'operator', 'interaction', 'user', 'mistake', 'transparency']):
        return 'Human-Interaction'
        
    return 'Other'

# --- Apply Mappings ---
aiid['Autonomy_Category'] = aiid['Autonomy Level'].apply(map_autonomy)
aiid['Failure_Category'] = aiid['Known AI Technical Failure'].apply(map_failure)

# --- Filter for Analysis ---
analysis_df = aiid[
    (aiid['Autonomy_Category'].notna()) &
    (aiid['Failure_Category'].isin(['Robustness', 'Specification', 'Human-Interaction']))
].copy()

print(f"\nData points after filtering: {len(analysis_df)}")
print("Autonomy distribution:")
print(analysis_df['Autonomy_Category'].value_counts())
print("Failure distribution:")
print(analysis_df['Failure_Category'].value_counts())

if len(analysis_df) < 5:
    print("Not enough data to perform Chi-Square test.")
else:
    # --- Cross-Tabulation ---
    ct = pd.crosstab(analysis_df['Autonomy_Category'], analysis_df['Failure_Category'])
    print("\n--- Cross-Tabulation (Counts) ---")
    print(ct)

    # --- Chi-Square Test ---
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # --- Visualization ---
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    print("\n--- Row Proportions (Normalized) ---")
    print(ct_norm)

    # Plot
    ax = ct_norm.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Distribution of Failure Modes by Autonomy Level', fontsize=14)
    plt.xlabel('Autonomy Level', fontsize=12)
    plt.ylabel('Proportion of Incidents', fontsize=12)
    plt.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # --- Hypothesis Check ---
    # H1: High Autonomy -> Specification
    # H2: Low Autonomy -> Human-Interaction
    
    try:
        high_spec = ct_norm.loc['High Autonomy', 'Specification'] if 'Specification' in ct_norm.columns else 0
        low_hci = ct_norm.loc['Low Autonomy', 'Human-Interaction'] if 'Human-Interaction' in ct_norm.columns else 0
        
        print(f"\nHigh Autonomy -> Specification: {high_spec:.1%}")
        print(f"Low Autonomy -> Human-Interaction: {low_hci:.1%}")
        
        if high_spec > 0.3 and low_hci > 0.3:
             print("Result: Hypothesis Supported (Trends match expectations)")
        else:
             print("Result: Hypothesis Not Supported (Trends do not match expectations)")
    except Exception as e:
        print(f"Could not validate hypothesis completely: {e}")