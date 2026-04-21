import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

def analyze_sectoral_ethics():
    # 1. Load Dataset
    filename = 'astalabs_discovery_all_data.csv'
    paths = [filename, '../' + filename]
    file_path = next((p for p in paths if os.path.exists(p)), None)

    if not file_path:
        print("Error: Dataset file not found.")
        return

    print(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Failed to load csv: {e}")
        return

    # Clean column names just in case
    df.columns = df.columns.str.strip()

    # Filter for EO 13960 Scored subset
    eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 subset shape: {eo_df.shape}")

    # 2. Define Sector Categories
    def map_sector(val):
        if pd.isna(val):
            return None
        topic = str(val).lower()
        if 'health' in topic:
            return 'Health'
        # Check for LE keywords
        le_keywords = ['law enforcement', 'justice', 'security', 'police', 'surveillance']
        if any(kw in topic for kw in le_keywords):
            return 'Law Enforcement/Justice'
        return None

    eo_df['sector_group'] = eo_df['8_topic_area'].apply(map_sector)
    
    # Filter for relevant sectors
    sector_df = eo_df[eo_df['sector_group'].notna()].copy()
    print("\nSector Distribution:")
    print(sector_df['sector_group'].value_counts())

    # 3. Analyze Target Variable: 62_disparity_mitigation
    target_col = '62_disparity_mitigation'
    if target_col not in sector_df.columns:
        print(f"\nError: Column '{target_col}' not found. Available columns: {list(sector_df.columns)}")
        return

    # Debug: Check unique values to ensure correct parsing
    unique_vals = sector_df[target_col].unique()
    print(f"\nUnique values in '{target_col}':")
    # Convert to string to print safely, taking first 10 if too many
    print(np.array2string(np.array(unique_vals, dtype=str))[:500])

    # Binarize
    # We look for explicit affirmation. Usually 'Yes' or descriptions starting with 'Yes'.
    def parse_compliance(val):
        if pd.isna(val):
            return 0
        s = str(val).lower().strip()
        # Check for common positive indicators
        if s.startswith('yes') or s == 'true' or s == '1':
            return 1
        return 0

    sector_df['has_mitigation'] = sector_df[target_col].apply(parse_compliance)

    # 4. Generate Statistics
    print("\n--- Analysis Results ---")
    
    # Contingency Table
    ct = pd.crosstab(sector_df['sector_group'], sector_df['has_mitigation'])
    
    # Ensure 0 and 1 columns exist
    for c in [0, 1]:
        if c not in ct.columns:
            ct[c] = 0
    
    # Reorder to [Yes, No] for readability and OR calculation
    ct = ct[[1, 0]]
    print("\nContingency Table (1=Yes, 0=No):")
    print(ct)

    # Calculate Rates
    rates = sector_df.groupby('sector_group')['has_mitigation'].mean() * 100
    print("\nCompliance Rates (%):")
    print(rates)

    # Check for sufficient data for Chi-Square
    # If 'Yes' column is all zeros, we can't run standard Chi-Square
    total_yes = ct[1].sum()
    if total_yes == 0:
        print("\nStatistical Test Skipped: Zero compliance found in both sectors.")
        print("Odds Ratio: Undefined (0/0 type scenario).")
        print("Conclusion: No evidence of disparity mitigation controls found in either sector.")
    else:
        # Chi-Square
        try:
            chi2, p, dof, expected = chi2_contingency(ct)
            print(f"\nChi-Square Statistic: {chi2:.4f}")
            print(f"P-value: {p:.4e}")
            
            # Odds Ratio
            # OR = (Health_Yes / Health_No) / (LE_Yes / LE_No)
            h_yes = ct.loc['Health', 1]
            h_no = ct.loc['Health', 0]
            le_yes = ct.loc['Law Enforcement/Justice', 1]
            le_no = ct.loc['Law Enforcement/Justice', 0]
            
            # Haldane-Anscombe correction if any cell is zero
            if (h_no * le_yes) == 0:
                print("\n(Using Haldane-Anscombe correction for Odds Ratio due to zero cell)")
                h_yes += 0.5
                h_no += 0.5
                le_yes += 0.5
                le_no += 0.5
            
            or_val = (h_yes * le_no) / (h_no * le_yes)
            print(f"Odds Ratio (Health vs LE): {or_val:.4f}")
            
        except Exception as e:
            print(f"\nStatistical test failed: {e}")

if __name__ == "__main__":
    analyze_sectoral_ethics()