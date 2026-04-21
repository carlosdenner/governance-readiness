import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded {len(aiid_df)} AIID incidents.")

# --- Step 1: Map Autonomy Level ---
# Based on previous exploration: 'Autonomy Level' contains 'Autonomy1', 'Autonomy2', 'Autonomy3'
autonomy_col = 'Autonomy Level' if 'Autonomy Level' in aiid_df.columns else 'autonomy'

def map_autonomy(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    # High Autonomy
    if 'autonomy3' in val_str or 'high' in val_str or 'full' in val_str:
        return 'High'
    # Low Autonomy
    if 'autonomy1' in val_str or 'autonomy2' in val_str or 'low' in val_str or 'human' in val_str:
        return 'Low'
    return np.nan

aiid_df['autonomy_category'] = aiid_df[autonomy_col].apply(map_autonomy)
print("\nAutonomy Distribution:")
print(aiid_df['autonomy_category'].value_counts(dropna=False))

# --- Step 2: Map Physical Harm (Text Analysis) ---
# We need to find the text column. Common names: 'description', 'summary', 'text'.
text_candidates = ['description', 'summary', 'Description', 'Summary', 'Text', 'incident_description']
text_col = next((c for c in text_candidates if c in aiid_df.columns), None)

print(f"\nUsing text column for harm analysis: {text_col}")

def map_physical_harm(row):
    # keywords for physical harm
    keywords = ['death', 'dead', 'kill', 'injury', 'injured', 'hurt', 'physical', 
                'collision', 'crash', 'accident', 'safety', 'burn', 'medical', 'hospital']
    
    text_content = ""
    if text_col and pd.notna(row[text_col]):
        text_content += str(row[text_col]).lower()
    
    # Also check 'Tangible Harm' or 'Harm Domain' for specific keywords if they exist
    if 'Tangible Harm' in row and pd.notna(row['Tangible Harm']):
        text_content += " " + str(row['Tangible Harm']).lower()
        
    if any(k in text_content for k in keywords):
        return 'Physical'
    return 'Non-Physical'

aiid_df['harm_category'] = aiid_df.apply(map_physical_harm, axis=1)
print("\nHarm Distribution:")
print(aiid_df['harm_category'].value_counts(dropna=False))

# --- Step 3: Statistical Analysis ---
analysis_df = aiid_df.dropna(subset=['autonomy_category'])
print(f"\nFinal analysis set size: {len(analysis_df)}")

if len(analysis_df) > 0:
    # Contingency Table
    contingency_table = pd.crosstab(analysis_df['autonomy_category'], analysis_df['harm_category'])
    print("\n--- Contingency Table (Autonomy vs Harm Type) ---")
    print(contingency_table)

    # Chi-Square
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Odds Ratio Calculation
    # OR = (High_Phys / High_NonPhys) / (Low_Phys / Low_NonPhys)
    try:
        if 'Physical' not in contingency_table.columns:
            print("\n'Physical' harm category missing from table. Cannot calculate Odds Ratio.")
        else:
            high_phys = contingency_table.loc['High', 'Physical']
            high_non = contingency_table.loc['High', 'Non-Physical']
            low_phys = contingency_table.loc['Low', 'Physical']
            low_non = contingency_table.loc['Low', 'Non-Physical']
            
            # Smoothing for zeros
            if high_non == 0 or low_phys == 0 or high_phys == 0 or low_non == 0:
                print("\n(Adding smoothing constant for zero cells)")
                high_phys += 0.5; high_non += 0.5; low_phys += 0.5; low_non += 0.5
            
            odds_ratio = (high_phys * low_non) / (high_non * low_phys)
            print(f"\nOdds Ratio (High Autonomy -> Physical Harm): {odds_ratio:.4f}")
    except KeyError:
        print("\nMissing categories for Odds Ratio calculation.")

    # Plot
    contingency_table.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])
    plt.title('Harm Type by Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Count')
    plt.legend(title='Harm Type')
    plt.tight_layout()
    plt.show()
else:
    print("\nNo data available for analysis.")
