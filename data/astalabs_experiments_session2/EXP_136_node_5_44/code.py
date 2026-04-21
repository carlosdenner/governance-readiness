import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

def run_experiment():
    try:
        print("Starting Temporal Severity Escalation Experiment...")
        
        # 1. Load Dataset
        filename = 'astalabs_discovery_all_data.csv'
        if not os.path.exists(filename):
            if os.path.exists(f'../{filename}'):
                filename = f'../{filename}'
            else:
                print("Error: Dataset not found.")
                return

        df = pd.read_csv(filename, low_memory=False)
        
        # 2. Filter for AIID incidents
        aiid = df[df['source_table'] == 'aiid_incidents'].copy()
        print(f"AIID Incidents loaded: {len(aiid)}")

        # 3. Process Dates
        aiid['date'] = pd.to_datetime(aiid['date'], errors='coerce')
        aiid = aiid.dropna(subset=['date'])
        aiid['year'] = aiid['date'].dt.year
        
        # Define Periods
        # Pre-2021 (<= 2020) and Post-2020 (>= 2021)
        aiid['Period'] = aiid['year'].apply(lambda x: 'Post-2020' if x >= 2021 else 'Pre-2021')
        
        # 4. Process Severity (Using schema found in debug)
        # High: 'AI tangible harm event'
        # Low: 'AI tangible harm near-miss', 'AI tangible harm issue', 'none'
        sev_col = 'AI Harm Level'
        
        def map_severity(val):
            if pd.isna(val):
                return None
            val = str(val).strip()
            if val == 'AI tangible harm event':
                return 'High_Severity'
            elif val in ['AI tangible harm near-miss', 'AI tangible harm issue', 'none']:
                return 'Low_Severity'
            return None

        aiid['Severity_Class'] = aiid[sev_col].apply(map_severity)
        
        # Filter for valid analysis rows
        analysis_df = aiid.dropna(subset=['Severity_Class']).copy()
        
        print(f"Records included in analysis: {len(analysis_df)}")
        print("Distribution of Severity Classes:")
        print(analysis_df['Severity_Class'].value_counts())

        # 5. Contingency Table
        ct = pd.crosstab(analysis_df['Period'], analysis_df['Severity_Class'])
        # Ensure consistent order
        ct = ct.reindex(index=['Pre-2021', 'Post-2020'], columns=['Low_Severity', 'High_Severity'], fill_value=0)
        
        print("\n--- Contingency Table ---")
        print(ct)

        # 6. Chi-Square Test
        chi2, p, dof, expected = chi2_contingency(ct)
        print(f"\nChi-Square Statistic: {chi2:.4f}")
        print(f"P-Value: {p:.4f}")
        
        # Interpretation
        if p < 0.05:
            print("Result: Statistically significant difference found.")
        else:
            print("Result: No statistically significant difference found.")

        # 7. Proportions
        props = pd.crosstab(analysis_df['Period'], analysis_df['Severity_Class'], normalize='index') * 100
        print("\n--- Proportions (%) ---")
        print(props.round(2))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_experiment()