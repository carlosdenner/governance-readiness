import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

print("--- Data Inspection ---")
# Columns identified from previous steps
autonomy_col = 'Autonomy Level'
tangible_col = 'Tangible Harm'
intangible_col1 = 'Harm Distribution Basis'
intangible_col2 = 'Special Interest Intangible Harm'

# --- Step 1: Categorize Autonomy ---
# Autonomy3 -> High
# Autonomy1, Autonomy2 -> Low
# Unclear -> Drop
def map_autonomy(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    if 'autonomy3' in val_str:
        return 'High Autonomy'
    elif 'autonomy1' in val_str or 'autonomy2' in val_str:
        return 'Low Autonomy'
    return None

aiid_df['Autonomy_Bin'] = aiid_df[autonomy_col].apply(map_autonomy)

# --- Step 2: Categorize Harm ---
# Tangible: 'tangible harm definitively occurred' OR 'imminent risk'
# Intangible: 'no tangible harm' AND (Intangible cols are populated/Yes)

def map_harm(row):
    t_val = str(row[tangible_col]).lower() if pd.notna(row[tangible_col]) else ''
    
    # Check Tangible
    if 'definitively occurred' in t_val or 'imminent risk' in t_val:
        return 'Tangible/Safety'
    
    # Check Intangible
    # If tangible is explicitly 'no' or 'issue' (non-imminent), check for intangible markers
    # We check if the intangible columns have meaningful content (not nan, not 'no')
    i1_val = str(row[intangible_col1]).lower() if pd.notna(row[intangible_col1]) else ''
    i2_val = str(row[intangible_col2]).lower() if pd.notna(row[intangible_col2]) else ''
    
    has_intangible = False
    if i1_val and i1_val not in ['nan', 'no', 'none', 'unclear']:
        has_intangible = True
    if i2_val and i2_val not in ['nan', 'no', 'none', 'unclear']:
        has_intangible = True
        
    if has_intangible:
        return 'Intangible/Societal'
        
    return None

aiid_df['Harm_Bin'] = aiid_df.apply(map_harm, axis=1)

# Drop unmapped rows
analysis_df = aiid_df.dropna(subset=['Autonomy_Bin', 'Harm_Bin'])

print(f"\nRows valid for analysis: {len(analysis_df)}")
print("Autonomy counts:\n", analysis_df['Autonomy_Bin'].value_counts())
print("Harm counts:\n", analysis_df['Harm_Bin'].value_counts())

if len(analysis_df) < 5:
    print("Insufficient data for statistical testing.")
else:
    # --- Step 3: Contingency Table & Stats ---
    contingency = pd.crosstab(analysis_df['Autonomy_Bin'], analysis_df['Harm_Bin'])
    print("\n--- Contingency Table ---")
    print(contingency)
    
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    # Fisher's Exact (if 2x2)
    if contingency.shape == (2, 2):
        odds, fisher_p = fisher_exact(contingency)
        print(f"Fisher's Exact P-value: {fisher_p:.4f}")
        print(f"Odds Ratio: {odds:.4f}")
    
    # Calculate Conditional Probabilities
    print("\nConditional Probabilities:")
    for autonomy in contingency.index:
        total = contingency.loc[autonomy].sum()
        if 'Tangible/Safety' in contingency.columns:
            tangible_count = contingency.loc[autonomy, 'Tangible/Safety']
            prob = tangible_count / total
            print(f"P(Tangible Harm | {autonomy}) = {tangible_count}/{total} ({prob:.2%})")
        else:
             print(f"P(Tangible Harm | {autonomy}) = 0/{total} (0.00%)")

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
    plt.title('Autonomy Level vs. Harm Type')
    plt.xlabel('Harm Category')
    plt.ylabel('Autonomy Level')
    plt.tight_layout()
    plt.show()