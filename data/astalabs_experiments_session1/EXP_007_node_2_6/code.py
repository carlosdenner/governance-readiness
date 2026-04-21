import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys

# [debug] Print python version and current working directory to understand environment
# import os
# print(sys.version)
# print(os.getcwd())

def run_experiment():
    try:
        # Load dataset
        # Note: User specified datasets are one level above
        file_path = '../step3_incident_coding.csv'
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            # Fallback if running in a different environment structure
            df = pd.read_csv('step3_incident_coding.csv')
            
        print(f"Loaded dataset with {len(df)} records.")
        
        # Check unique values in split column to ensure correct filtering
        print(f"Unique values in 'trust_integration_split': {df['trust_integration_split'].unique()}")
        
        # Standardize strings just in case
        df['split_norm'] = df['trust_integration_split'].astype(str).str.lower().str.strip()
        
        # Filter groups
        # Target groups: 'trust-dominant' and 'integration-dominant'
        trust_group = df[df['split_norm'] == 'trust-dominant']['technique_count']
        integration_group = df[df['split_norm'] == 'integration-dominant']['technique_count']
        
        print(f"\nSample sizes:\nTrust-Dominant: {len(trust_group)}\nIntegration-Dominant: {len(integration_group)}")
        
        # Check if we have enough data
        if len(trust_group) < 2 or len(integration_group) < 2:
            print("\nWARNING: Sample sizes are too small for reliable statistical testing.")
            print("Proceeding with available data for demonstration purposes.")

        # Calculate Summary Statistics
        stats_summary = pd.DataFrame({
            'Group': ['Trust-Dominant', 'Integration-Dominant'],
            'Count': [len(trust_group), len(integration_group)],
            'Mean': [trust_group.mean(), integration_group.mean()],
            'Median': [trust_group.median(), integration_group.median()],
            'Std': [trust_group.std(), integration_group.std()]
        })
        print("\n=== Summary Statistics ===")
        print(stats_summary.to_string(index=False))
        
        # Statistical Test
        # Using Mann-Whitney U test as sample sizes are small and normality is not guaranteed
        if len(trust_group) > 0 and len(integration_group) > 0:
            u_stat, p_val = stats.mannwhitneyu(trust_group, integration_group, alternative='two-sided')
            print("\n=== Statistical Test Results (Mann-Whitney U) ===")
            print(f"U-statistic: {u_stat}")
            print(f"P-value: {p_val:.4f}")
            alpha = 0.05
            if p_val < alpha:
                print("Result: Statistically significant difference (p < 0.05)")
            else:
                print("Result: No statistically significant difference (p >= 0.05)")
        else:
            print("\nCannot perform statistical test due to empty groups.")

        # Visualization
        if len(trust_group) > 0 or len(integration_group) > 0:
            plt.figure(figsize=(10, 6))
            # Prepare data for boxplot
            data_to_plot = []
            labels = []
            
            if len(trust_group) > 0: 
                data_to_plot.append(trust_group)
                labels.append(f'Trust-Dominant\n(n={len(trust_group)})')
            if len(integration_group) > 0:
                data_to_plot.append(integration_group)
                labels.append(f'Integration-Dominant\n(n={len(integration_group)})')
            
            plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
            plt.title('Distribution of Technique Counts by Competency Split')
            plt.ylabel('Technique Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_experiment()