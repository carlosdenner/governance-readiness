import pandas as pd
import numpy as np
import scipy.stats as stats
import os

def run_experiment():
    print("Starting Data Governance Cascade experiment (Attempt 2)...")
    
    # Load dataset
    file_name = 'astalabs_discovery_all_data.csv'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, low_memory=False)
    elif os.path.exists('../' + file_name):
        df = pd.read_csv('../' + file_name, low_memory=False)
    else:
        print("Error: Dataset not found.")
        return

    # Filter for EO 13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 records loaded: {len(df_eo)}")

    col_catalog = '31_data_catalog'
    col_docs = '34_data_docs'

    # 1. Clean Data Catalog (Q31)
    # Map 'Yes' (case-insensitive) to 1, else 0
    def clean_catalog(val):
        if pd.isna(val):
            return 0
        s = str(val).strip().lower()
        if 'yes' in s:
            return 1
        return 0
    
    df_eo['has_catalog'] = df_eo[col_catalog].apply(clean_catalog)

    # 2. Clean Data Documentation (Q34)
    # Map specific maturity levels to 1
    def clean_docs(val):
        if pd.isna(val):
            return 0
        s = str(val).strip().lower()
        # Positive indicators based on maturity model text
        if any(x in s for x in ['partially completed', 'is complete', 'widely available']):
            return 1
        # Negative indicators (explicitly checked for validation, though default is 0)
        # 'missing', 'not available' -> 0
        return 0

    df_eo['has_docs'] = df_eo[col_docs].apply(clean_docs)

    # Verification of mapping
    print("\n--- Mapping Verification ---")
    print("Catalog (Q31) Sample Mappings:")
    print(df_eo[[col_catalog, 'has_catalog']].drop_duplicates().head(5))
    print("\nDocs (Q34) Sample Mappings:")
    # Show unique strings and their mapping to ensure correctness
    unique_docs = df_eo[[col_docs, 'has_docs']].drop_duplicates()
    # Print first few chars of the long strings for readability
    for idx, row in unique_docs.iterrows():
        raw_val = str(row[col_docs])[:60] + "..."
        print(f"Raw: {raw_val:<65} -> Mapped: {row['has_docs']}")

    # 3. Contingency Table
    contingency = pd.crosstab(df_eo['has_catalog'], df_eo['has_docs'])
    print("\n--- Contingency Table (Rows=Catalog, Cols=Docs) ---")
    print(contingency)
    
    # Extract counts
    # contingency structure: 
    # col   0    1
    # row
    # 0     TN   FN
    # 1     FP   TP
    try:
        tn = contingency.loc[0, 0]
        fp = contingency.loc[0, 1] if 1 in contingency.columns else 0
        fn = contingency.loc[1, 0] if 1 in contingency.index else 0
        tp = contingency.loc[1, 1] if 1 in contingency.index and 1 in contingency.columns else 0
    except KeyError:
        print("Error creating full contingency table (missing classes).")
        return

    total = tn + fp + fn + tp
    n_catalog = fn + tp
    n_docs = fp + tp

    print(f"\nTotal Cases: {total}")
    print(f"Has Catalog: {n_catalog} ({n_catalog/total:.1%})")
    print(f"Has Docs:    {n_docs} ({n_docs/total:.1%})")

    # 4. Statistical Analysis
    # Conditional Probability P(Docs | Catalog)
    if n_catalog > 0:
        p_docs_given_catalog = tp / n_catalog
        p_docs_baseline = n_docs / total
        lift = p_docs_given_catalog / p_docs_baseline if p_docs_baseline > 0 else 0
        
        print(f"\nConditional Probability P(Docs|Catalog): {p_docs_given_catalog:.4f}")
        print(f"Baseline Probability P(Docs):          {p_docs_baseline:.4f}")
        print(f"Lift (Strength of Dependency):         {lift:.2f}x")
    
    # Chi-Squared and Phi
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    # Phi Coefficient = sqrt(chi2 / n)
    phi = np.sqrt(chi2 / total)
    
    print(f"\n--- Statistical Tests ---")
    print(f"Chi-Squared Statistic: {chi2:.4f}")
    print(f"P-Value:               {p_val:.4e}")
    print(f"Phi Coefficient:       {phi:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    if p_val < 0.05:
        print("Statistically significant relationship detected.")
        if phi > 0.5: print("Effect size: Strong")
        elif phi > 0.3: print("Effect size: Moderate")
        elif phi > 0.1: print("Effect size: Weak")
        else: print("Effect size: Negligible")
    else:
        print("No statistically significant relationship detected.")

if __name__ == "__main__":
    run_experiment()