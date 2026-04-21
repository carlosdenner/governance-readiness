import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# [debug] Check current directory and file existence
print(f"Current working directory: {os.getcwd()}")
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    # Fallback if running in same directory
    file_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from: {file_path}")

try:
    # Load the dataset
    df = pd.read_csv(file_path, low_memory=False)
    
    # Filter for 'eo13960_scored'
    eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Filtered EO 13960 records: {len(eo_df)}")
    
    # Inspect unique values to determine mapping
    print("\nUnique values in '16_dev_stage':")
    print(eo_df['16_dev_stage'].value_counts(dropna=False).head(10))
    
    print("\nUnique values in '52_impact_assessment':")
    print(eo_df['52_impact_assessment'].value_counts(dropna=False))
    
    # Define mapping logic
    def map_stage(stage):
        if pd.isna(stage):
            return None
        s = str(stage).lower()
        # Operation: Use, Maintenance, Operation
        if any(x in s for x in ['use', 'maintenance', 'operation', 'production']):
            return 'Operation'
        # Pre-Operation: Planning, Development, Acquisition
        elif any(x in s for x in ['planning', 'development', 'acquisition', 'pilot', 'test']):
            return 'Pre-Operation'
        return None

    def map_compliance(val):
        # Map 'Yes' to 1, others to 0
        if pd.isna(val):
            return 0
        if 'yes' in str(val).lower():
            return 1
        return 0

    # Apply mappings
    eo_df['stage_group'] = eo_df['16_dev_stage'].apply(map_stage)
    eo_df['is_compliant'] = eo_df['52_impact_assessment'].apply(map_compliance)
    
    # Filter for valid stages
    analysis_df = eo_df[eo_df['stage_group'].notna()].copy()
    
    # Calculate statistics
    summary = analysis_df.groupby('stage_group')['is_compliant'].agg(['count', 'sum', 'mean'])
    summary.columns = ['Total Systems', 'Compliant Systems', 'Compliance Rate']
    
    print("\n--- Summary Statistics ---")
    print(summary)
    
    # Perform Statistical Test (Chi-Square)
    # Create contingency table
    contingency = pd.crosstab(analysis_df['stage_group'], analysis_df['is_compliant'])
    print("\n--- Contingency Table ---")
    print(contingency)
    
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")
    
    if p < 0.05:
        print("Result: The difference in compliance rates is statistically significant.")
    else:
        print("Result: No statistically significant difference found.")

    # Generate Bar Chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(summary.index, summary['Compliance Rate'], color=['#1f77b4', '#ff7f0e'])
    
    plt.title('Impact Assessment Compliance: Operation vs Pre-Operation')
    plt.ylabel('Compliance Rate (Proportion)')
    plt.xlabel('Lifecycle Stage')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
