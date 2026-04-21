import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

def run_experiment():
    # Load dataset
    # Based on previous successful attempts, the file is in the current directory.
    file_path = 'astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    # Filter for ATLAS cases
    # Source table identifier from metadata: 'atlas_cases'
    atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
    
    if atlas_df.empty:
        print("No ATLAS cases found in the dataset.")
        return

    print(f"Loaded {len(atlas_df)} ATLAS cases.")

    # Pre-processing
    # Ensure relevant columns are strings and handle NaNs
    # In the metadata, column 92 is 'tactics' and 93 is 'techniques'
    # Depending on the sparse structure, column names might be preserved or just indices.
    # The previous exploration showed 'tactics' and 'techniques' columns exist.
    
    # Verify columns exist
    if 'tactics' not in atlas_df.columns or 'techniques' not in atlas_df.columns:
        print("Columns 'tactics' or 'techniques' not found.")
        print("Available columns:", atlas_df.columns.tolist())
        return

    atlas_df['tactics'] = atlas_df['tactics'].fillna('').astype(str)
    atlas_df['techniques'] = atlas_df['techniques'].fillna('').astype(str)

    # 1. Define 'Has_Impact'
    # Check if 'Impact' is in the tactics list
    atlas_df['Has_Impact'] = atlas_df['tactics'].str.contains('Impact', case=False)

    # 2. Calculate 'Complexity' (Count of techniques)
    def count_techniques(val):
        if not val or val.strip() == '':
            return 0
        # Techniques are often separated by commas or semicolons in such datasets
        normalized = val.replace(';', ',')
        items = [x.strip() for x in normalized.split(',') if x.strip()]
        return len(set(items)) # Unique items

    atlas_df['Complexity'] = atlas_df['techniques'].apply(count_techniques)

    # Separate groups
    impact_complexity = atlas_df[atlas_df['Has_Impact'] == True]['Complexity']
    no_impact_complexity = atlas_df[atlas_df['Has_Impact'] == False]['Complexity']

    # Descriptive Statistics
    print("\n--- Descriptive Statistics ---")
    print(f"Group 'Has Impact' (n={len(impact_complexity)}):")
    if len(impact_complexity) > 0:
        print(f"  Mean Complexity: {impact_complexity.mean():.2f}")
        print(f"  Median Complexity: {impact_complexity.median()}")
        print(f"  Std Dev: {impact_complexity.std():.2f}")
    else:
        print("  No data.")
    
    print(f"\nGroup 'No Impact' (n={len(no_impact_complexity)}):")
    if len(no_impact_complexity) > 0:
        print(f"  Mean Complexity: {no_impact_complexity.mean():.2f}")
        print(f"  Median Complexity: {no_impact_complexity.median()}")
        print(f"  Std Dev: {no_impact_complexity.std():.2f}")
    else:
        print("  No data.")

    # Statistical Testing
    if len(impact_complexity) > 1 and len(no_impact_complexity) > 1:
        # Using Mann-Whitney U test as technique counts are discrete and sample size is small
        u_stat, p_val = stats.mannwhitneyu(impact_complexity, no_impact_complexity, alternative='two-sided')
        
        # T-test for comparison (Welch's)
        t_stat, p_val_t = stats.ttest_ind(impact_complexity, no_impact_complexity, equal_var=False)

        print("\n--- Statistical Test Results ---")
        print(f"Mann-Whitney U Test: U={u_stat}, p-value={p_val:.4f}")
        print(f"Welch's T-Test: t={t_stat:.4f}, p-value={p_val_t:.4f}")

        alpha = 0.05
        if p_val < alpha:
            print("\nResult: Statistically SIGNIFICANT difference in complexity found.")
        else:
            print("\nResult: NO statistically significant difference in complexity found.")
            
        # Visualization
        try:
            plt.figure(figsize=(10, 6))
            data_to_plot = [no_impact_complexity, impact_complexity]
            labels = ['No Impact Tactic', 'Has Impact Tactic']
            
            plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
            plt.title('Adversarial Tactic Complexity by Impact Status')
            plt.ylabel('Count of Techniques Used')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        except Exception as e:
            print(f"Plotting error: {e}")
            
    else:
        print("\nInsufficient data for statistical testing.")

if __name__ == "__main__":
    run_experiment()