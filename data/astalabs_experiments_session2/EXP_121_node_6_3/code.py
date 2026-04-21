import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import os

def run_experiment():
    # Load dataset
    file_path = 'astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        file_path = '../astalabs_discovery_all_data.csv'
        df = pd.read_csv(file_path, low_memory=False)

    # Filter for AIID incidents
    df_incidents = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents loaded: {len(df_incidents)} rows")

    # Dynamic Column Identification
    cols = df_incidents.columns
    sector_col = next((c for c in cols if 'Sector' in c and 'Deployment' in c), None)
    autonomy_col = next((c for c in cols if 'Autonomy' in c and 'Level' in c), None)

    print(f"Using Sector Column: {sector_col}")
    print(f"Using Autonomy Column: {autonomy_col}")

    if not sector_col or not autonomy_col:
        print("Critical columns missing. Aborting.")
        return

    # 1. Map Sector
    def map_sector(val):
        if pd.isna(val): return None
        val_str = str(val).lower().strip()
        if 'transportation' in val_str:
            return 'Transportation'
        if any(x in val_str for x in ['government', 'public', 'administration']):
            return 'Government'
        return None

    df_incidents['mapped_sector'] = df_incidents[sector_col].apply(map_sector)
    
    # Filter to relevant sectors
    df_filtered = df_incidents[df_incidents['mapped_sector'].notna()].copy()
    print(f"Rows after sector filtering: {len(df_filtered)}")
    print(f"Sector counts:\n{df_filtered['mapped_sector'].value_counts()}")

    # 2. Map Autonomy
    # Mapping based on observed values: ['Autonomy1', 'Autonomy2', 'Autonomy3', 'unclear']
    # Assumption: 1-2 = Low, 3+ = High
    def map_autonomy(val):
        if pd.isna(val): return None
        val_str = str(val).lower().strip()
        
        # Specific dataset tags
        if 'autonomy1' in val_str or 'autonomy2' in val_str:
            return 'Low'
        if 'autonomy3' in val_str or 'autonomy4' in val_str or 'autonomy5' in val_str:
            return 'High'
        
        # Generic text fallback
        if 'high' in val_str: return 'High'
        if 'low' in val_str: return 'Low'
        
        return None

    df_filtered['mapped_autonomy'] = df_filtered[autonomy_col].apply(map_autonomy)
    
    # Filter valid autonomy rows
    df_final = df_filtered[df_filtered['mapped_autonomy'].notna()].copy()
    print(f"Rows after autonomy filtering: {len(df_final)}")
    
    if len(df_final) == 0:
        print("No data available for analysis.")
        return

    # 3. Contingency Table
    contingency = pd.crosstab(df_final['mapped_sector'], df_final['mapped_autonomy'])
    print("\n--- Contingency Table (Sector vs Autonomy) ---")
    print(contingency)

    # 4. Chi-square Test
    # Ensure we have data in the table to run the test
    if contingency.size >= 4:
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print("\n--- Chi-square Test Results ---")
        print(f"Chi-square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.6e}")
        
        # Interpretation
        alpha = 0.05
        if p < alpha:
            print("Result: REJECT Null Hypothesis. Significant difference found.")
            
            # Calculate High Autonomy percentages
            row_props = pd.crosstab(df_final['mapped_sector'], df_final['mapped_autonomy'], normalize='index') * 100
            print("\nProportions (%):")
            print(row_props.round(2))
            
            try:
                trans_high = row_props.loc['Transportation', 'High']
            except KeyError:
                trans_high = 0
            
            try:
                gov_high = row_props.loc['Government', 'High']
            except KeyError:
                gov_high = 0
            
            print(f"\nAnalysis: Transportation High Autonomy Rate: {trans_high:.1f}%")
            print(f"Analysis: Government High Autonomy Rate: {gov_high:.1f}%")
            
            if trans_high > gov_high:
                print("Conclusion: Hypothesis SUPPORTED. Transportation has a higher rate of high-autonomy incidents.")
            else:
                print("Conclusion: Hypothesis CONTRADICTED. Government has a higher rate (or equal) of high-autonomy incidents.")
        else:
            print("Result: FAIL TO REJECT Null Hypothesis. No significant difference in autonomy levels between sectors.")
    else:
        print("Contingency table too small for Chi-square test.")

if __name__ == "__main__":
    run_experiment()