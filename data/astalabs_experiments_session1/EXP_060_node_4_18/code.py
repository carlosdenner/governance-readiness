import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# [debug]
print("Current Working Directory:", os.getcwd())
print("Files in parent directory:", os.listdir('..'))

# Define file path based on instruction
file_path = '../step3_tactic_frequency.csv'

# Robustness check in case environment differs from instruction
if not os.path.exists(file_path):
    file_path = 'step3_tactic_frequency.csv'

try:
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Filter data by bundle
    integration_data = df[df['bundle'] == 'Integration Readiness']['incident_count']
    trust_data = df[df['bundle'] == 'Trust Readiness']['incident_count']

    # Calculate Descriptive Statistics
    int_mean = integration_data.mean()
    int_median = integration_data.median()
    int_std = integration_data.std()
    int_n = len(integration_data)

    trust_mean = trust_data.mean()
    trust_median = trust_data.median()
    trust_std = trust_data.std()
    trust_n = len(trust_data)

    print("\n=== Descriptive Statistics ===")
    print(f"Integration Readiness (n={int_n}): Mean={int_mean:.2f}, Median={int_median:.2f}, Std={int_std:.2f}")
    print(f"Trust Readiness       (n={trust_n}): Mean={trust_mean:.2f}, Median={trust_median:.2f}, Std={trust_std:.2f}")

    # Perform Mann-Whitney U Test
    # Using 'two-sided' to detect any difference, though hypothesis implies Integration > Trust
    u_stat, p_val = stats.mannwhitneyu(integration_data, trust_data, alternative='two-sided')

    print("\n=== Mann-Whitney U Test Results ===")
    print(f"U-statistic: {u_stat}")
    print(f"P-value: {p_val:.5f}")
    
    if p_val < 0.05:
        print("Result: Statistically Significant Difference")
    else:
        print("Result: No Statistically Significant Difference")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.boxplot([integration_data, trust_data], labels=['Integration Readiness', 'Trust Readiness'], patch_artist=True)
    plt.title('Incident Count Frequency by Bundle')
    plt.ylabel('Incident Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")