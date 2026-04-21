import pandas as pd
import numpy as np
import scipy.stats as stats
import os

def run_experiment():
    # Determine file path
    filename = 'astalabs_discovery_all_data.csv'
    if not os.path.exists(filename):
        # Fallback to parent directory if not found in current
        if os.path.exists(f'../{filename}'):
            filename = f'../{filename}'
        else:
            print(f"Error: {filename} not found in current ({os.getcwd()}) or parent directory.")
            return

    print(f"Loading dataset from: {filename}")
    try:
        df = pd.read_csv(filename, low_memory=False)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # Filter for 'eo13960_scored' source
    eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Filtered EO13960 records: {len(eo_df)}")

    if eo_df.empty:
        print("No data available for analysis.")
        return

    # Define columns
    col_code = '38_code_access'
    col_appeal = '65_appeal_process'

    # Check columns
    if col_code not in eo_df.columns or col_appeal not in eo_df.columns:
        print(f"Required columns missing. Available: {eo_df.columns.tolist()}")
        return

    # Inspect raw values to ensure correct mapping
    print(f"\n--- Value Counts: {col_code} ---")
    print(eo_df[col_code].value_counts(dropna=False).head(10))
    print(f"\n--- Value Counts: {col_appeal} ---")
    print(eo_df[col_appeal].value_counts(dropna=False).head(10))

    # Data Cleaning / Binary Mapping
    # Hypothesis: Transparency (Code Access) -> Accountability (Appeal Process)
    
    # Map 38_code_access: 1 if indicates public/open availability, 0 otherwise.
    # Common affirmative values: 'Yes', 'Open Source', 'Public', 'Available on GitHub', etc.
    def map_transparency(val):
        s = str(val).lower()
        if any(x in s for x in ['yes', 'open', 'public', 'github', 'available']):
            return 1
        return 0

    # Map 65_appeal_process: 1 if 'Yes', 0 otherwise.
    def map_accountability(val):
        s = str(val).lower()
        if 'yes' in s:
            return 1
        return 0

    eo_df['is_transparent'] = eo_df[col_code].apply(map_transparency)
    eo_df['has_appeal'] = eo_df[col_appeal].apply(map_accountability)

    # Create Contingency Table
    contingency = pd.crosstab(
        eo_df['is_transparent'], 
        eo_df['has_appeal'],
        rownames=['Code Transparency'],
        colnames=['Appeal Process']
    )
    
    print("\n--- Contingency Table ---")
    print(contingency)

    # Statistical Test (Chi-Square)
    # We need a 2x2 table. If dimensions are smaller (e.g., constant values), warn user.
    if contingency.shape != (2, 2):
        print("\nWarning: Contingency table is not 2x2 (lack of variance in one variable).")
        # Try to run chi2 anyway if valid, or just print warning
        if contingency.size >= 2:
             chi2, p, dof, expected = stats.chi2_contingency(contingency)
             print(f"Chi-square: {chi2:.4f}, p-value: {p:.5f}")
        return

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    phi = np.sqrt(chi2 / n)

    print("\n--- Statistical Results ---")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"p-value: {p:.5f}")
    print(f"Phi Coefficient: {phi:.4f}")

    # Calculate probabilities for interpretation
    # P(Appeal | Transparent)
    p_appeal_given_trans = contingency.loc[1, 1] / contingency.loc[1].sum() if contingency.loc[1].sum() > 0 else 0
    # P(Appeal | Not Transparent)
    p_appeal_given_opaque = contingency.loc[0, 1] / contingency.loc[0].sum() if contingency.loc[0].sum() > 0 else 0

    print("\n--- Interpretation ---")
    print(f"Probability of Appeal Process given Open Code: {p_appeal_given_trans:.2%}")
    print(f"Probability of Appeal Process given Closed Code: {p_appeal_given_opaque:.2%}")

if __name__ == '__main__':
    run_experiment()