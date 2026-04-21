import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Define file path (try parent directory first, then current)
filename = 'step3_coverage_map.csv'
file_path = f"../{filename}"
if not os.path.exists(file_path):
    file_path = filename

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path)
    
    # filter for specific bundles
    trust_data = df[df['bundle'] == 'Trust Readiness']['incident_count']
    integration_data = df[df['bundle'] == 'Integration Readiness']['incident_count']
    
    print("\n--- Descriptive Statistics ---")
    print(f"Trust Readiness: n={len(trust_data)}, Mean={trust_data.mean():.2f}, Std={trust_data.std():.2f}")
    print(f"Integration Readiness: n={len(integration_data)}, Mean={integration_data.mean():.2f}, Std={integration_data.std():.2f}")
    
    # Check for normality to decide on test
    print("\n--- Normality Check (Shapiro-Wilk) ---")
    shapiro_trust = stats.shapiro(trust_data)
    shapiro_integration = stats.shapiro(integration_data)
    print(f"Trust: W={shapiro_trust.statistic:.4f}, p={shapiro_trust.pvalue:.4f}")
    print(f"Integration: W={shapiro_integration.statistic:.4f}, p={shapiro_integration.pvalue:.4f}")
    
    alpha = 0.05
    if shapiro_trust.pvalue > alpha and shapiro_integration.pvalue > alpha:
        print("\nData appears normal. Performing Independent Samples T-test (Welch's).")
        stat, p_value = stats.ttest_ind(trust_data, integration_data, equal_var=False)
        test_type = "Welch's T-test"
    else:
        print("\nData deviates from normality. Performing Mann-Whitney U test.")
        stat, p_value = stats.mannwhitneyu(trust_data, integration_data, alternative='two-sided')
        test_type = "Mann-Whitney U Test"
        
    print(f"\n--- Hypothesis Test Results ({test_type}) ---")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: Statistically Significant Difference")
    else:
        print("Result: No Statistically Significant Difference")
        
    # Visualization
    plt.figure(figsize=(10, 6))
    # Create a list of data to plot
    data_to_plot = [trust_data, integration_data]
    
    # Create the boxplot
    box = plt.boxplot(data_to_plot, patch_artist=True, labels=['Trust Readiness', 'Integration Readiness'])
    
    # Colors
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    # Add individual points (jittered)
    for i, data in enumerate(data_to_plot):
        y = data
        x = np.random.normal(1 + i, 0.04, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.5)

    plt.title('Incident Coverage Distribution by Competency Bundle')
    plt.ylabel('Incident Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")