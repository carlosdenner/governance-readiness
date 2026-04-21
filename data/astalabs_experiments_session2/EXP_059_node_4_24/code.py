import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

# Attempt to locate the file
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, '../' + filename, '/content/' + filename]
data_path = None

for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    # If not found, try to list directories to debug, but for now assuming it's in current based on Exp 1
    data_path = filename

print(f"Loading dataset from: {data_path}")

try:
    df = pd.read_csv(data_path, low_memory=False)
    
    # Filter for EO13960 scored data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 subset shape: {df_eo.shape}")

    # Check columns
    stage_col = '16_dev_stage'
    test_col = '53_real_world_testing'
    
    if stage_col not in df_eo.columns or test_col not in df_eo.columns:
        print(f"Columns missing. Available: {df_eo.columns.tolist()[:10]}...")
    else:
        # Clean and Normalize Stage
        # Inspect unique values first
        print("\n--- Raw Lifecycle Stages ---")
        print(df_eo[stage_col].value_counts(dropna=False))
        
        def map_lifecycle(stage):
            s = str(stage).lower()
            if 'operation' in s or 'maintenance' in s:
                return 'Operational'
            elif 'development' in s or 'implementation' in s or 'plan' in s:
                return 'Development'
            else:
                return 'Other'
        
        df_eo['lifecycle_group'] = df_eo[stage_col].apply(map_lifecycle)
        
        # Filter for only Operational and Development
        df_analysis = df_eo[df_eo['lifecycle_group'].isin(['Operational', 'Development'])].copy()
        
        print("\n--- Analysis Groups ---")
        print(df_analysis['lifecycle_group'].value_counts())
        
        # Clean and Normalize Testing Evidence
        print("\n--- Raw Testing Values ---")
        print(df_analysis[test_col].value_counts(dropna=False).head(10))
        
        # Strict 'Yes' criteria for evidence. Anything else (No, N/A, blank) is treated as lack of positive evidence.
        df_analysis['has_evidence'] = df_analysis[test_col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        
        # Calculate Rates
        results = df_analysis.groupby('lifecycle_group')['has_evidence'].agg(['count', 'sum', 'mean'])
        results['pct'] = results['mean'] * 100
        
        print("\n--- Testing Documentation Rates ---")
        print(results)
        
        # Statistical Test (Chi-Square)
        # Contingency Table
        contingency = pd.crosstab(df_analysis['lifecycle_group'], df_analysis['has_evidence'])
        print("\n--- Contingency Table (0=No Evidence, 1=Yes Evidence) ---")
        print(contingency)
        
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        print(f"\nChi-Square Statistic: {chi2:.4f}")
        print(f"P-Value: {p:.4e}")
        
        # Hypothesis Check
        op_rate = results.loc['Operational', 'mean']
        dev_rate = results.loc['Development', 'mean']
        
        print(f"\nOperational Rate: {op_rate:.2%}")
        print(f"Development Rate: {dev_rate:.2%}")
        
        if p < 0.05:
            print("Result: Statistically significant difference.")
            if op_rate < dev_rate:
                print("Hypothesis SUPPORTED: Operational systems have significantly lower documentation rates.")
            else:
                print("Hypothesis REJECTED: Operational systems do not have lower rates (direction inverted).")
        else:
            print("Result: No statistically significant difference.")
            print("Hypothesis REJECTED.")
            
except Exception as e:
    print(f"Execution failed: {e}")
