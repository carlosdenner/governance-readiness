import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"EO 13960 Dataset Size: {len(eo_df)}")

# --- 1. Variable Construction ---

# Construct 'is_public_service'
# Logic: If '26_public_service' has content (and isn't 'No'), it's a public service.
# If it's NaN or 'No', it's Internal.
eo_df['26_public_service'] = eo_df['26_public_service'].astype(str).replace('nan', np.nan)

def classify_service(val):
    if pd.isna(val):
        return False
    if val.lower().strip() == 'no':
        return False
    # If it has substantial text, it's a description of the service -> True
    if len(val) > 2:
        return True
    return False

eo_df['is_public_service'] = eo_df['26_public_service'].apply(classify_service)

print("\nConstructed 'is_public_service':")
print(eo_df['is_public_service'].value_counts())

# Construct 'is_commercial'
# Logic: Based on column 10_commercial_ai.
# "None of the above." -> Custom/Other (False)
# Specific use cases -> Commercial (True)
# NaN -> Assume Custom/Other (False) for now, or exclude. Let's assume False to be conservative.

def classify_commercial(val):
    if pd.isna(val):
        return False
    s = str(val).strip()
    if "None of the above" in s:
        return False
    return True

eo_df['is_commercial'] = eo_df['10_commercial_ai'].apply(classify_commercial)

print("\nConstructed 'is_commercial':")
print(eo_df['is_commercial'].value_counts())

# --- 2. Contingency Analysis ---

# Create Crosstab
contingency = pd.crosstab(eo_df['is_public_service'], eo_df['is_commercial'])

# Labeling for clarity (Printing raw first to avoid index errors)
print("\n--- Raw Contingency Table ---")
print(contingency)

# Check if we have a 2x2 matrix
if contingency.shape == (2, 2):
    contingency.index = ['Internal/Admin', 'Public Service']
    contingency.columns = ['Custom/Gov-Built', 'Commercial/COTS']
    print("\n--- Labeled Contingency Table ---")
    print(contingency)
    
    # --- 3. Statistical Testing ---
    
    # Chi-Square
    chi2, p, dof, ex = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Odds Ratio (Fisher Exact)
    # Table: [[Internal_Custom, Internal_Comm], [Public_Custom, Public_Comm]]
    # OR > 1 means Public is MORE likely to be Commercial
    # OR < 1 means Public is LESS likely to be Commercial
    # We want OR of Commercial for Public vs Internal
    # Odds_Public = Comm / Custom
    # Odds_Internal = Comm / Custom
    # OR = Odds_Public / Odds_Internal
    
    # Use Fisher Exact for precision
    # fisher_exact expects [[a, b], [c, d]]
    # We want to check association between Row 2 (Public) and Col 2 (Commercial)
    # Let's align it: 
    #              Comm   Custom
    # Public       a      b
    # Internal     c      d
    
    # Current table:
    #              Custom  Comm
    # Internal     A       B
    # Public       C       D
    
    # Re-arranging for the specific hypothesis test:
    # Rows: Public, Internal
    # Cols: Commercial, Custom
    
    # Public_Comm
    pc = contingency.loc['Public Service', 'Commercial/COTS']
    # Public_Custom
    p_cust = contingency.loc['Public Service', 'Custom/Gov-Built']
    # Internal_Comm
    ic = contingency.loc['Internal/Admin', 'Commercial/COTS']
    # Internal_Custom
    i_cust = contingency.loc['Internal/Admin', 'Custom/Gov-Built']
    
    obs = [[pc, p_cust], [ic, i_cust]]
    
    odds_r, p_val_fisher = fisher_exact(obs)
    
    print(f"\nFisher Exact Odds Ratio: {odds_r:.4f}")
    print(f"Fisher P-value: {p_val_fisher:.4e}")
    
    if odds_r < 1:
        print(f"Result: Public-facing systems are {1/odds_r:.2f}x LESS likely to use Commercial AI.")
    else:
        print(f"Result: Public-facing systems are {odds_r:.2f}x MORE likely to use Commercial AI.")

else:
    print("\nError: Contingency table is not 2x2. Cannot perform Odds Ratio analysis.")
    print("Check variable construction logic.")

# --- 4. Visualization ---
if contingency.shape == (2, 2):
    # Calculate percentages
    props = contingency.div(contingency.sum(axis=1), axis=0)
    
    ax = props.plot(kind='bar', stacked=True, color=['lightgray', 'steelblue'], figsize=(8, 6))
    plt.title('Commercial AI Adoption: Public Service vs Internal')
    plt.ylabel('Proportion')
    plt.xlabel('Service Type')
    plt.legend(title='Procurement', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()