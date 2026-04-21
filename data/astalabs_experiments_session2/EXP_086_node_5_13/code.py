import pandas as pd
import numpy as np
import scipy.stats as stats
import os

def run_experiment():
    print("Starting experiment: Lifecycle Documentation Debt (Attempt 2)...")
    
    # 1. Load Data
    filename = 'astalabs_discovery_all_data.csv'
    if os.path.exists(filename):
        filepath = filename
    elif os.path.exists(os.path.join('..', filename)):
        filepath = os.path.join('..', filename)
    else:
        print(f"Error: {filename} not found.")
        return

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # 2. Filter for EO13960
    if 'source_table' not in df.columns:
        print("Column 'source_table' not found.")
        return
        
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 Scored rows: {len(df_eo)}")

    # 3. Identify Columns (Dynamic Search)
    cols = df_eo.columns
    dev_col = next((c for c in cols if 'dev_stage' in c), None)
    doc_col = next((c for c in cols if 'data_docs' in c), None)
    
    if not dev_col or not doc_col:
        print(f"Critical columns missing. Found dev: {dev_col}, doc: {doc_col}")
        return

    print(f"Using Dev Stage Column: '{dev_col}'")
    print(f"Using Documentation Column: '{doc_col}'")

    # 4. Map Lifecycle Stage
    # Operation: 'Operation', 'Maintenance', 'In production', 'In mission'
    # Development: 'Acquisition', 'Development', 'Initiated', 'Implementation', 'Planned'
    def map_lifecycle(val):
        if pd.isna(val): return np.nan
        s = str(val).lower()
        if any(x in s for x in ['operation', 'maintenance', 'in production', 'in mission']):
            return 'Operation'
        if any(x in s for x in ['development', 'acquisition', 'initiated', 'implementation', 'planned']):
            return 'Development'
        return np.nan # Exclude Retired or unclear

    df_eo['lifecycle_group'] = df_eo[dev_col].apply(map_lifecycle)
    
    # 5. Map Documentation Status (Parsing Descriptions)
    # Hypothesis specifies "complete" documentation.
    # We will map "Complete" and "Widely available" to 1.
    # We will map "Missing", "Not available", and "Partially" to 0 (Incomplete/Debt).
    def map_docs(val):
        if pd.isna(val): return np.nan
        s = str(val).lower()
        if 'complete' in s or 'widely available' in s:
            # Check for partial negation
            if 'partially' in s:
                return 0
            return 1
        if 'missing' in s or 'not available' in s or 'partially' in s:
            return 0
        return 0 # Default to 0 if unclear but text exists

    df_eo['has_complete_docs'] = df_eo[doc_col].apply(map_docs)

    # 6. Prepare Analysis DataFrame
    # Drop NaNs in either target column to analyze only valid comparisons
    df_analysis = df_eo.dropna(subset=['lifecycle_group', 'has_complete_docs'])
    
    print(f"\nTotal rows for analysis (valid stage & doc status): {len(df_analysis)}")
    
    # 7. Generate Contingency Table
    ct = pd.crosstab(df_analysis['lifecycle_group'], df_analysis['has_complete_docs'])
    # Ensure shape
    for i in [0, 1]:
        if i not in ct.columns: ct[i] = 0
    
    # Sort columns to [0, 1] -> [Incomplete, Complete]
    ct = ct[[0, 1]]
    ct.columns = ['Incomplete Docs', 'Complete Docs']
    
    print("\n--- Contingency Table ---")
    print(ct)
    
    # 8. Statistical Tests
    chi2, p, dof, ex = stats.chi2_contingency(ct)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    # Odds Ratio
    # OR = (Op_Complete * Dev_Incomplete) / (Op_Incomplete * Dev_Complete)
    # Logic: Odds of being Complete in Operation vs Development
    
    op_comp = ct.loc['Operation', 'Complete Docs']
    op_inc = ct.loc['Operation', 'Incomplete Docs']
    dev_comp = ct.loc['Development', 'Complete Docs']
    dev_inc = ct.loc['Development', 'Incomplete Docs']
    
    if op_inc * dev_comp == 0:
        odds_ratio = np.inf 
    else:
        odds_ratio = (op_comp * dev_inc) / (op_inc * dev_comp)
        
    print(f"Odds Ratio (Complete Docs given Operation): {odds_ratio:.4f}")
    
    # Calculate Percentages
    op_rate = op_comp / (op_comp + op_inc) if (op_comp + op_inc) > 0 else 0
    dev_rate = dev_comp / (dev_comp + dev_inc) if (dev_comp + dev_inc) > 0 else 0
    
    print(f"\nDocumentation Completion Rate (Operation): {op_rate:.2%}")
    print(f"Documentation Completion Rate (Development): {dev_rate:.2%}")
    
    # 9. Conclusion
    print("\n--- Conclusion ---")
    if p < 0.05:
        print("Result: Statistically Significant.")
        if op_rate < dev_rate:
            print("Hypothesis SUPPORTED: Operational systems are significantly less likely to have complete documentation.")
        else:
            print("Hypothesis REJECTED: Operational systems are significantly MORE likely to have complete documentation.")
    else:
        print("Result: Not Statistically Significant.")
        print("Hypothesis REJECTED due to lack of significance.")

if __name__ == "__main__":
    run_experiment()