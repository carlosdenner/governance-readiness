import pandas as pd
import numpy as np
import scipy.stats as stats
import os

def clean_binary_text(val):
    if pd.isna(val):
        return 0
    text = str(val).strip().lower()
    if not text:
        return 0
    
    # List of terms indicating absence of the control
    negatives = [
        'no', 'none', 'n/a', 'not applicable', 'tbd', 'unknown', 
        'false', '0', 'nan', 'not established', 'not currently'
    ]
    if text in negatives:
        return 0
    
    # explicit negative phrases
    if text.startswith('no ') or text.startswith('not '):
        return 0
        
    # Default to 1 (presence of evidence/description)
    return 1

def main():
    filename = 'astalabs_discovery_all_data.csv'
    
    # specific check for file existence
    if not os.path.exists(filename):
        # fallback to parent if current not found, though previous error suggests parent is wrong
        if os.path.exists(f'../{filename}'):
            filename = f'../{filename}'
        else:
            print(f"Error: {filename} not found in current or parent directory.")
            return
            
    print(f"Loading dataset from {filename}...")
    try:
        df = pd.read_csv(filename, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Filter for EO 13960 Scored
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Filtered EO 13960 data: {len(df_eo)} rows")

    # Columns
    col_impact = '52_impact_assessment'
    col_monitor = '56_monitor_postdeploy'
    
    # Apply binary classification
    df_eo['has_impact_assessment'] = df_eo[col_impact].apply(clean_binary_text)
    df_eo['has_monitoring'] = df_eo[col_monitor].apply(clean_binary_text)
    
    # Descriptive stats
    print(f"\nImpact Assessment (1=Yes, 0=No):\n{df_eo['has_impact_assessment'].value_counts()}")
    print(f"\nPost-Deployment Monitoring (1=Yes, 0=No):\n{df_eo['has_monitoring'].value_counts()}")
    
    # Contingency Table
    contingency = pd.crosstab(df_eo['has_impact_assessment'], df_eo['has_monitoring'])
    print("\nContingency Table (Rows: Assessment, Cols: Monitoring):")
    print(contingency)
    
    # Check if we have a valid 2x2
    if contingency.shape != (2, 2):
        print("\nWarning: Contingency table is not 2x2 (likely missing 0s or 1s in one dimension). Stats may be degenerate.")

    # Statistics
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    
    # Pearson correlation (equivalent to Phi for binary variables, preserving sign)
    correlation = df_eo['has_impact_assessment'].corr(df_eo['has_monitoring'])
    
    print(f"\n--- Statistics ---")
    print(f"Chi-Square: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Phi Coefficient (Pearson r): {correlation:.4f}")
    
    # Interpretation
    print("\n--- Conclusion ---")
    if p < 0.05:
        if correlation > 0.1:
            print("Result: Significant POSITIVE correlation. (Hypothesis Rejected)")
            print("Interpretation: Agencies performing impact assessments ARE likely to perform monitoring.")
        elif correlation < -0.1:
            print("Result: Significant NEGATIVE correlation. (Hypothesis Supported)")
            print("Interpretation: Inverse relationship found.")
        else:
            print("Result: Statistically significant but negligible effect size.")
    else:
        print("Result: No significant correlation found. (Hypothesis Supported)")
        print("Interpretation: Assessment and Monitoring appear to be decoupled.")

if __name__ == "__main__":
    main()