import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

def run_experiment():
    print("Starting Privacy Shield Paradox Experiment...\n")
    
    # 1. Load dataset
    filename = 'astalabs_discovery_all_data.csv'
    if os.path.exists(filename):
        filepath = filename
    elif os.path.exists(f'../{filename}'):
        filepath = f'../{filename}'
    else:
        print(f"Error: {filename} not found.")
        return

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # 2. Filter for 'eo13960_scored'
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 Subset Shape: {df_eo.shape}")

    if df_eo.empty:
        print("No data in EO 13960 subset.")
        return

    # Helper function for robust string conversion
    def robust_contains_yes(val):
        s = str(val).lower().strip()
        # Check if it starts with yes or is just yes (handling potential verbose answers)
        # The prompt says: 1 if 'Yes', 0 otherwise. 
        # Usually these fields contain "Yes", "No", "Not Applicable" or verbose explanations starting with Yes.
        # We will check if "yes" is in the string to be inclusive, or starts with it.
        if 'yes' in s:
            return 1
        return 0

    # 3. Create binary variable 'has_pii'
    if '29_contains_pii' not in df_eo.columns:
        print("Error: Column '29_contains_pii' not found.")
        return
    
    df_eo['has_pii'] = df_eo['29_contains_pii'].apply(robust_contains_yes)

    # 4. Create binary variable 'has_impact_assessment'
    if '52_impact_assessment' not in df_eo.columns:
        print("Error: Column '52_impact_assessment' not found.")
        return

    df_eo['has_impact_assessment'] = df_eo['52_impact_assessment'].apply(robust_contains_yes)

    # 5. Generate Contingency Table
    # Rows: Has PII (0, 1)
    # Cols: Has Impact Assessment (0, 1)
    contingency_table = pd.crosstab(
        df_eo['has_pii'], 
        df_eo['has_impact_assessment'], 
        rownames=['Has PII'], 
        colnames=['Has Impact Assessment']
    )

    print("\n--- Contingency Table ---")
    print(contingency_table)
    
    # Ensure table is 2x2 for consistent OR calc
    # Reindex to ensure all keys exist
    contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    # 6. Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Odds Ratio
    # OR = (a*d) / (b*c)
    # a = PII=0, IA=0
    # b = PII=0, IA=1
    # c = PII=1, IA=0
    # d = PII=1, IA=1
    
    a = contingency_table.loc[0, 0]
    b = contingency_table.loc[0, 1]
    c = contingency_table.loc[1, 0]
    d = contingency_table.loc[1, 1]
    
    if b * c > 0:
        odds_ratio = (a * d) / (b * c)
    else:
        odds_ratio = float('inf')

    # 7. Print Results
    print("\n--- Statistical Results ---")
    print(f"Chi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    
    # Interpretation
    print("\n--- Interpretation ---")
    if p < 0.05:
        print("Result: Statistically Significant")
        if odds_ratio > 1:
            print("Conclusion: Systems handling PII are MORE likely to have an Impact Assessment.")
        else:
            print("Conclusion: Systems handling PII are LESS likely to have an Impact Assessment.")
    else:
        print("Result: Not Statistically Significant")
        print("Conclusion: No significant association between PII and Impact Assessments.")

    # Descriptive Stats
    total_n = len(df_eo)
    pii_n = df_eo['has_pii'].sum()
    ia_n = df_eo['has_impact_assessment'].sum()
    
    print(f"\nTotal Analyzed: {total_n}")
    print(f"Systems with PII: {pii_n} ({pii_n/total_n:.1%})")
    print(f"Systems with Impact Assessment: {ia_n} ({ia_n/total_n:.1%})")
    
    if pii_n > 0:
        ia_rate_pii = d / pii_n
        print(f"Impact Assessment Rate (Given PII): {ia_rate_pii:.1%}")
    if (total_n - pii_n) > 0:
        ia_rate_no_pii = b / (total_n - pii_n)
        print(f"Impact Assessment Rate (Given No PII): {ia_rate_no_pii:.1%}")

if __name__ == "__main__":
    run_experiment()