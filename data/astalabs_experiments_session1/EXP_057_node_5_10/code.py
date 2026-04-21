import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_experiment():
    # 1. Load the dataset
    # Checking parent directory first as per instructions
    filename = 'step2_crosswalk_matrix.csv'
    if os.path.exists(f'../{filename}'):
        filepath = f'../{filename}'
    elif os.path.exists(filename):
        filepath = filename
    else:
        print(f"Error: {filename} not found in current or parent directory.")
        return

    print(f"Loading dataset from: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return
    
    # 2. Identify Control Columns
    # Known metadata columns: 'req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement'
    # All subsequent columns are architecture controls
    metadata_cols = {'req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement'}
    control_cols = [c for c in df.columns if c not in metadata_cols]
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of control columns identified: {len(control_cols)}")

    # 3. Calculate Control Count
    # Cells contain 'X' (or text) if mapped, empty/NaN otherwise.
    def get_control_count(row):
        count = 0
        for col in control_cols:
            val = row[col]
            if pd.notna(val) and str(val).strip() != '':
                count += 1
        return count

    df['control_count'] = df.apply(get_control_count, axis=1)

    # 4. Filter and Group by Source
    # We define two groups: 'EU AI Act' and 'NIST AI RMF' (grouping all NIST variants)
    print("\nOriginal Source Counts:")
    print(df['source'].value_counts())

    def classify_source(s):
        s_str = str(s).upper()
        if 'EU AI ACT' in s_str:
            return 'EU AI Act'
        elif 'NIST' in s_str:
            return 'NIST AI RMF'
        return None

    df['framework_group'] = df['source'].apply(classify_source)
    
    # Remove rows that don't match these two groups (e.g., OWASP)
    df_filtered = df.dropna(subset=['framework_group'])
    
    print("\nGrouped Framework Counts:")
    print(df_filtered['framework_group'].value_counts())

    # 5. Statistical Analysis
    group_eu = df_filtered[df_filtered['framework_group'] == 'EU AI Act']['control_count']
    group_nist = df_filtered[df_filtered['framework_group'] == 'NIST AI RMF']['control_count']

    # Descriptive Statistics
    print("\n--- Descriptive Statistics (Control Count) ---")
    print(df_filtered.groupby('framework_group')['control_count'].describe())

    # Check if we have data in both groups
    if len(group_eu) < 2 or len(group_nist) < 2:
        print("\nInsufficient data for t-test.")
    else:
        # Welch's t-test (equal_var=False)
        t_stat, p_val = stats.ttest_ind(group_eu, group_nist, equal_var=False)
        
        print("\n--- Welch's T-Test Results ---")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_val:.4f}")
        
        alpha = 0.05
        if p_val < alpha:
            print("Result: Statistically significant difference found.")
        else:
            print("Result: No statistically significant difference found.")

    # 6. Visualization
    if len(group_eu) > 0 and len(group_nist) > 0:
        plt.figure(figsize=(8, 6))
        data = [group_eu, group_nist]
        labels = ['EU AI Act', 'NIST AI RMF']
        
        plt.boxplot(data, labels=labels, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'))
        
        plt.title('Architectural Control Density: EU AI Act vs NIST AI RMF')
        plt.ylabel('Number of Mapped Controls')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.show()

if __name__ == "__main__":
    run_experiment()