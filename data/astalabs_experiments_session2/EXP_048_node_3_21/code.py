import pandas as pd
import numpy as np
import scipy.stats as stats

def run_experiment():
    print("Starting 'Paper Tiger' Assessment Experiment (Attempt 2)...")
    
    # 1. Load Data
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("File not found in parent directory. Trying current directory.")
        try:
            df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        except FileNotFoundError:
            print("Dataset not found.")
            return

    # 2. Filter for EO 13960 Scored
    eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 Scored subset shape: {eo_df.shape}")

    # 3. Identify Columns Dynamically
    cols = eo_df.columns.tolist()
    
    def get_col_by_keyword(keywords):
        matches = [c for c in cols if all(k.lower() in c.lower() for k in keywords)]
        if matches:
            return matches[0]
        return None

    col_assessment = get_col_by_keyword(['52', 'impact', 'assessment'])
    col_adverse = get_col_by_keyword(['61', 'adverse', 'impact'])
    col_mitigation = get_col_by_keyword(['62', 'disparity', 'mitigation']) 
    # Fallback if 62 not found by number
    if not col_mitigation:
        col_mitigation = get_col_by_keyword(['disparity', 'mitigation'])

    print(f"Using columns:\n  Assessment: {col_assessment}\n  Adverse: {col_adverse}\n  Mitigation: {col_mitigation}")

    if not (col_assessment and col_adverse and col_mitigation):
        print("Critical columns missing. Aborting.")
        return

    # 4. Filter for Impact Assessment == YES
    # Normalize to string, lower, strip
    eo_df['assess_norm'] = eo_df[col_assessment].astype(str).str.lower().str.strip()
    
    target_vals = ['yes', 'true', '1']
    analyzed_df = eo_df[eo_df['assess_norm'].isin(target_vals)].copy()
    
    print(f"Rows with Impact Assessment found: {len(analyzed_df)}")

    if len(analyzed_df) < 5:
        print("Warning: Very few data points. Statistical tests may be invalid.")

    # 5. Create Binary Variables (Fixing previous bug)
    def make_binary(val):
        # Convert to string, lower case, strip whitespace
        s = str(val).lower().strip()
        # Check against truthy values
        return 1 if s in ['yes', 'true', '1'] else 0

    analyzed_df['risk_found'] = analyzed_df[col_adverse].apply(make_binary)
    analyzed_df['action_taken'] = analyzed_df[col_mitigation].apply(make_binary)

    # 6. Analysis
    # Contingency Table
    contingency = pd.crosstab(analyzed_df['risk_found'], analyzed_df['action_taken'])
    # Ensure 2x2 shape
    contingency = contingency.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    
    contingency.index = ['No Adverse Impact (0)', 'Adverse Impact Found (1)']
    contingency.columns = ['No Mitigation (0)', 'Mitigation Taken (1)']

    print("\nContingency Table:")
    print(contingency)

    # Phi Coefficient
    phi = analyzed_df['risk_found'].corr(analyzed_df['action_taken'])
    print(f"\nPhi Coefficient (Correlation): {phi:.4f}")

    # Chi-Square Test
    # Check if we have enough data variance
    if contingency.values.sum() == 0:
         print("No data populated.")
    elif (contingency.values == 0).all():
         print("Contingency table empty.")
    else:
        try:
            chi2, p, dof, ex = stats.chi2_contingency(contingency)
            print(f"Chi-Square Statistic: {chi2:.4f}")
            print(f"P-Value: {p:.4f}")

            # Conclusion
            print("\n--- Conclusion ---")
            if p > 0.05:
                print("Result: No significant association found (P > 0.05).")
                print("Supports 'Paper Tiger' hypothesis: Finding risks does not reliably predict mitigation actions.")
            else:
                print("Result: Significant association found (P <= 0.05).")
                print("Rejects 'Paper Tiger' hypothesis: Finding risks is associated with taking mitigation actions.")
        except Exception as e:
            print(f"Statistical test failed: {e}")

if __name__ == "__main__":
    run_experiment()