import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import os

# Load the dataset
# Try potential paths
paths = ['../astalabs_discovery_all_data.csv', 'astalabs_discovery_all_data.csv']
file_path = None
for p in paths:
    if os.path.exists(p):
        file_path = p
        break

if file_path is None:
    print("Error: Dataset file not found.")
else:
    try:
        df = pd.read_csv(file_path, low_memory=False)
        
        # Filter for EO 13960 scored data
        eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
        print(f"Loaded {len(eo_df)} rows from EO 13960 dataset.")

        # --- Step 1: Categorize Lifecycle Stage (16_dev_stage) ---
        # Inspect unique values to ensure correct mapping
        unique_stages = eo_df['16_dev_stage'].unique()
        print(f"\nUnique values in '16_dev_stage' (first 10): {unique_stages[:10]}")

        def map_lifecycle(stage):
            if pd.isna(stage):
                return None
            stage_str = str(stage).lower()
            
            ops_keywords = ['operation', 'maintenance', 'use', 'implemented', 'production']
            dev_keywords = ['development', 'planning', 'acquisition', 'initiation', 'design', 'pilot']
            
            # Check for Ops keywords
            if any(k in stage_str for k in ops_keywords):
                return 'Legacy/Ops'
            # Check for Dev keywords
            elif any(k in stage_str for k in dev_keywords):
                return 'New/Dev'
            else:
                return 'Other'

        eo_df['lifecycle_group'] = eo_df['16_dev_stage'].apply(map_lifecycle)

        # Filter out 'Other' or None
        valid_lifecycle_df = eo_df[eo_df['lifecycle_group'].isin(['Legacy/Ops', 'New/Dev'])].copy()

        print(f"\nRows after filtering for valid lifecycle stages: {len(valid_lifecycle_df)}")
        print(valid_lifecycle_df['lifecycle_group'].value_counts())

        # --- Step 2: Categorize Impact Assessment (52_impact_assessment) ---
        # Inspect unique values
        unique_assessments = valid_lifecycle_df['52_impact_assessment'].unique()
        print(f"\nUnique values in '52_impact_assessment': {unique_assessments}")

        def map_assessment(val):
            if pd.isna(val):
                return 'No'
            val_str = str(val).lower()
            # Strict 'yes' check or explicit positive indicator
            if val_str == 'yes' or 'completed' in val_str:
                return 'Yes'
            return 'No'

        valid_lifecycle_df['has_assessment'] = valid_lifecycle_df['52_impact_assessment'].apply(map_assessment)

        # --- Step 3: Analysis ---

        # Contingency Table
        contingency = pd.crosstab(valid_lifecycle_df['lifecycle_group'], valid_lifecycle_df['has_assessment'])
        # Ensure columns exist even if one category is empty
        if 'Yes' not in contingency.columns:
            contingency['Yes'] = 0
        if 'No' not in contingency.columns:
            contingency['No'] = 0
            
        contingency = contingency[['No', 'Yes']] # Reorder
        
        print("\nContingency Table (Count):")
        print(contingency)

        # Calculate percentages
        results = contingency.copy()
        results['Total'] = results['No'] + results['Yes']
        results['Compliance Rate (%)'] = (results['Yes'] / results['Total']) * 100

        print("\nCompliance Rates by Lifecycle Stage:")
        print(results[['Total', 'Compliance Rate (%)']])

        # --- Step 4: Statistical Test (Chi-Square) ---
        chi2, p, dof, expected = chi2_contingency(contingency)

        print(f"\nChi-Square Test Results:")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"P-value: {p:.6f}")

        alpha = 0.05
        if p < alpha:
            print("Result: Statistically Significant.")
            rate_dev = results.loc['New/Dev', 'Compliance Rate (%)']
            rate_ops = results.loc['Legacy/Ops', 'Compliance Rate (%)']
            if rate_dev > rate_ops:
                print(f"Hypothesis Supported: New/Dev systems ({rate_dev:.1f}%) have higher compliance than Legacy/Ops ({rate_ops:.1f}%).")
            else:
                print(f"Hypothesis Contradicted: Legacy/Ops systems ({rate_ops:.1f}%) have higher compliance than New/Dev ({rate_dev:.1f}%).")
        else:
            print("Result: Not Statistically Significant. No evidence of difference in compliance.")

        # Visualization
        if not results.empty:
            ax = results['Compliance Rate (%)'].plot(kind='bar', color=['skyblue', 'salmon'], figsize=(8, 6))
            plt.title('Impact Assessment Compliance by Lifecycle Stage')
            plt.ylabel('Compliance Rate (%)')
            plt.xlabel('Lifecycle Stage')
            plt.ylim(0, 100)
            
            for i, v in enumerate(results['Compliance Rate (%)']):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"An error occurred during execution: {e}")
