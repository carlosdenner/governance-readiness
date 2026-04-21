import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# 1. Load dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback for different directory structure if needed
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for 'aiid_incidents'
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)}")

# 3. Identify columns
# Normalize column names to find matches easily
aiid_df.columns = [c.strip() for c in aiid_df.columns]

tangible_col = next((c for c in aiid_df.columns if 'Tangible Harm' in c), None)
tech_fail_col = next((c for c in aiid_df.columns if 'Known AI Technical Failure' in c), None)

if not tangible_col or not tech_fail_col:
    print(f"Columns not found. Tangible: {tangible_col}, Tech: {tech_fail_col}")
    exit()

print(f"Using columns: '{tangible_col}' and '{tech_fail_col}'")

# 4. Define Mapping Logic based on observed values
# Observed values: 
# - 'tangible harm definitively occurred'
# - 'no tangible harm, near-miss, or issue'
# - 'non-imminent risk...'
# - 'imminent risk...'
# - NaN

def categorize_harm(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower().strip()
    
    # Exact string matching based on previous debug output
    if 'tangible harm definitively occurred' in val_str:
        return 'Tangible'
    elif 'no tangible harm' in val_str:
        return 'Intangible'
    else:
        return None # Exclude risks/near-misses/unclear to be strict about 'Harm Incidents'

aiid_df['harm_category'] = aiid_df[tangible_col].apply(categorize_harm)

# Filter out nulls (which includes NaNs and Risks)
analysis_df = aiid_df[aiid_df['harm_category'].notna()].copy()

print(f"\nRows available for analysis after filtering: {len(analysis_df)}")
print(analysis_df['harm_category'].value_counts())

# 5. Define Diagnosis Logic
# Check if 'Known AI Technical Failure' is populated with something meaningful
def categorize_diagnosis(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    # If it says 'unknown' explicitly, or is empty, it's not diagnosed.
    if val_str in ['', 'unknown', 'unspecified', 'nan']:
        return 0
    return 1

analysis_df['has_diagnosis'] = analysis_df[tech_fail_col].apply(categorize_diagnosis)

# 6. Statistical Analysis
if analysis_df.empty or len(analysis_df['harm_category'].unique()) < 2:
    print("Not enough data for comparison.")
else:
    # Contingency Table
    ct = pd.crosstab(analysis_df['harm_category'], analysis_df['has_diagnosis'])
    # Ensure both columns 0/1 exist for display
    if 0 not in ct.columns: ct[0] = 0
    if 1 not in ct.columns: ct[1] = 0
    ct = ct[[0, 1]]
    ct.columns = ['No Diagnosis', 'Diagnosis Present']
    
    print("\n--- Contingency Table ---")
    print(ct)

    # Chi-square Test
    chi2, p, dof, expected = chi2_contingency(ct)
    
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")

    # Rates
    rates = analysis_df.groupby('harm_category')['has_diagnosis'].mean()
    print("\n--- Diagnosis Rates ---")
    print(rates)

    # Visualization
    plt.figure(figsize=(8, 6))
    colors = ['#ff9999', '#66b3ff'] # Red for Intangible, Blue for Tangible
    ax = rates.plot(kind='bar', color=colors, edgecolor='black', rot=0)
    
    plt.title('Technical Failure Diagnosis Rate: Tangible vs Intangible Harm')
    plt.ylabel('Proportion with Identified Technical Failure')
    plt.xlabel('Harm Type')
    plt.ylim(0, 1.0)
    
    # Annotate bars
    for i, v in enumerate(rates):
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
