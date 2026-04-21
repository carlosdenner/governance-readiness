import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

def run_experiment():
    # 1. Load Data
    fname = 'astalabs_discovery_all_data.csv'
    fpath = f'../{fname}' if os.path.exists(f'../{fname}') else fname
    
    print("Loading data...")
    try:
        df = pd.read_csv(fpath, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return
    
    # Filter EO13960
    if 'source_table' not in df.columns:
        print("Column 'source_table' missing.")
        return

    eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 rows: {len(eo)}")
    
    # 2. Inspect Unique Values for Mapping
    topic_col = '8_topic_area'
    notice_col = '59_ai_notice'
    
    print(f"\nUnique values in '{topic_col}':")
    print(eo[topic_col].dropna().unique())
    
    print(f"\nUnique values in '{notice_col}':")
    unique_notices = eo[notice_col].dropna().unique()
    for val in unique_notices:
        print(f"  - {val}")

    # 3. Categorize Topic (IV)
    def categorize_topic(val):
        val_str = str(val).lower()
        # Surveillance / Security / Enforcement
        if any(x in val_str for x in ['law', 'justice', 'security', 'defense', 'intelligence', 'border', 'police', 'enforcement']):
            return 'Surveillance/Security'
        # Service / Benefits
        if any(x in val_str for x in ['health', 'service', 'benefit', 'education', 'transportation', 'housing', 'agriculture', 'energy', 'labor', 'veteran', 'environment', 'commerce']):
            return 'Service/Benefits'
        return 'Other'

    eo['Group'] = eo[topic_col].apply(categorize_topic)
    
    # 4. Clean Notice (DV)
    # Logic: If the field indicates a method of notice (Online, Email, In-person), it's Yes.
    # If it says 'None', 'N/A', or is empty, it's No.
    def clean_notice(val):
        if pd.isna(val):
            return np.nan # Treat missing as missing, or could be No. Let's exclude for now.
        val_str = str(val).lower().strip()
        
        # Negative indicators
        if any(x in val_str for x in ['none', 'n/a', 'not applicable', 'no notice']):
            return 'No'
            
        # Positive indicators (explicit methods)
        if any(x in val_str for x in ['online', 'email', 'in-person', 'person', 'web', 'mail', 'media', 'press', 'notification', 'posted']):
            return 'Yes'
            
        # Ambiguous 'Other' - check if it has content, but usually 'Other' implies some notice method was used but not listed.
        # However, let's be conservative. If it just says 'other', we might treat as Yes if we assume they selected 'Other' to describe a method.
        # Let's inspect 'Other' cases if they dominate, but generally 'Other' in a checkbox list implies 'Yes, via another method'.
        if 'other' in val_str:
            return 'Yes'
            
        return 'No' # Default fall-through

    eo['Notice_Provided'] = eo[notice_col].apply(clean_notice)
    
    # 5. Analysis
    # Filter for the two groups of interest
    valid = eo[
        (eo['Group'].isin(['Surveillance/Security', 'Service/Benefits'])) & 
        (eo['Notice_Provided'].notna())
    ].copy()
    
    print(f"\nValid rows for analysis: {len(valid)}")
    print("Group distribution:\n", valid['Group'].value_counts())
    print("Notice distribution:\n", valid['Notice_Provided'].value_counts())
    
    # Contingency Table
    ct = pd.crosstab(valid['Group'], valid['Notice_Provided'])
    print("\n--- Contingency Table (Counts) ---")
    print(ct)
    
    # Check for empty intersection
    if ct.size == 0 or 'Yes' not in ct.columns or 'No' not in ct.columns:
        print("\nError: Contingency table is missing columns (Yes/No) or is empty. Cannot run Chi-Square.")
        return

    # Percentages
    ct_pct = pd.crosstab(valid['Group'], valid['Notice_Provided'], normalize='index') * 100
    print("\n--- Contingency Table (Percentages) ---")
    print(ct_pct.round(2))
    
    # Chi-Square Test
    chi2, p, dof, exp = chi2_contingency(ct)
    print(f"\n--- Statistical Test Results ---")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Interpretation
    surv_rate = ct_pct.loc['Surveillance/Security', 'Yes']
    serv_rate = ct_pct.loc['Service/Benefits', 'Yes']
    
    print("\n--- Interpretation ---")
    print(f"Surveillance Notice Rate: {surv_rate:.1f}%")
    print(f"Service Notice Rate:      {serv_rate:.1f}%")
    
    if p < 0.05:
        if surv_rate < serv_rate:
            print("Significant: Surveillance provides LESS notice (Hypothesis Supported).")
        else:
            print("Significant: Surveillance provides MORE notice (Hypothesis Rejected).")
    else:
        print("Not Significant: No statistical difference found.")

if __name__ == "__main__":
    run_experiment()