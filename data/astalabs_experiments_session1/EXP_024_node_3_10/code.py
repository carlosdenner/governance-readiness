import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency, fisher_exact

# Helper to find file
def find_file(filename):
    paths = [filename, os.path.join('..', filename)]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

file_path = find_file('step3_incident_coding.csv')

if not file_path:
    print("Error: step3_incident_coding.csv not found.")
else:
    try:
        # 1. Load Data
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {df.shape}")

        # 2. Group 'failure_mode' into 'Prevention' vs 'Post-Breach'
        # Check unique values first
        print("\nUnique Failure Modes:", df['failure_mode'].unique())
        
        def group_failure(mode):
            m = str(mode).lower()
            if 'prevention' in m:
                return 'Prevention'
            elif 'detection' in m or 'response' in m:
                return 'Post-Breach'
            else:
                return 'Other'
        
        df['Failure_Phase'] = df['failure_mode'].apply(group_failure)
        
        # 3. Cross-tabulate with 'trust_integration_split'
        # Check unique values
        print("Unique Bundles:", df['trust_integration_split'].unique())
        
        # Create Contingency Table
        contingency = pd.crosstab(df['Failure_Phase'], df['trust_integration_split'])
        print("\n=== Contingency Table (Failure Phase vs Bundle) ===")
        print(contingency)
        
        # 4. Statistical Test
        # Determine which test to run based on table size and content
        # We are looking for association. Chi-Square is standard, Fisher for small samples.
        # Given the metadata warnings about skew, we likely have small cell counts.
        
        chi2, p, dof, expected = chi2_contingency(contingency)
        print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}\nP-value: {p:.4f}\nDegrees of Freedom: {dof}")
        print("Expected Frequencies:\n", expected)
        
        # If 2x2, run Fisher's Exact as well for robustness
        if contingency.shape == (2, 2):
            odds_ratio, p_fisher = fisher_exact(contingency)
            print(f"\nFisher's Exact Test P-value: {p_fisher:.4f}")

        # 5. Generate Heatmap
        plt.figure(figsize=(8, 5))
        # Use matplotlib directly to avoid seaborn dependency issues
        plt.imshow(contingency, cmap='Blues', aspect='auto')
        plt.colorbar(label='Incident Count')
        
        # Add labels
        cols = contingency.columns.tolist()
        rows = contingency.index.tolist()
        
        plt.xticks(range(len(cols)), cols, rotation=45)
        plt.yticks(range(len(rows)), rows)
        plt.title('Heatmap of Failure Phase vs. Competency Bundle')
        plt.xlabel('Competency Bundle')
        plt.ylabel('Failure Phase')
        
        # Add text annotations
        for i in range(len(rows)):
            for j in range(len(cols)):
                plt.text(j, i, contingency.iloc[i, j], ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")