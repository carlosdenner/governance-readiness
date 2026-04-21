import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# 1. Load the dataset
file_path = 'step3_coverage_map.csv'

if not os.path.exists(file_path):
    # Fallback to try finding it if the environment is weird, though previous steps suggest it is here.
    # This is just a safety check print.
    print(f"File {file_path} not found in {os.getcwd()}")
else:
    print(f"Loading {file_path}...")

df = pd.read_csv(file_path)

# 2. Group data and extract incident counts
integration_counts = df[df['bundle'] == 'Integration Readiness']['incident_count']
trust_counts = df[df['bundle'] == 'Trust Readiness']['incident_count']

# Calculate statistics
integration_mean = integration_counts.mean()
trust_mean = trust_counts.mean()

print(f"Integration Readiness (n={len(integration_counts)}): Mean Incident Count = {integration_mean:.2f}")
print(f"Trust Readiness (n={len(trust_counts)}): Mean Incident Count = {trust_mean:.2f}")
print(f"Integration Counts: {list(integration_counts)}")
print(f"Trust Counts: {list(trust_counts)}")

# 3. Perform Mann-Whitney U test
# We use 'two-sided' to detect any difference, though the hypothesis predicts Integration > Trust.
stat, p_value = stats.mannwhitneyu(integration_counts, trust_counts, alternative='two-sided')

print("\n=== Mann-Whitney U Test Results ===")
print(f"U-statistic: {stat}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant difference found.")
else:
    print("Result: No statistically significant difference found.")

# 4. Visualize
plt.figure(figsize=(8, 6))
bundles = ['Integration Readiness', 'Trust Readiness']
means = [integration_mean, trust_mean]

# Add error bars (standard error)
integration_sem = stats.sem(integration_counts)
trust_sem = stats.sem(trust_counts)
sems = [integration_sem, trust_sem]

# Plot
plt.bar(bundles, means, yerr=sems, capsize=10, color=['skyblue', 'lightgreen'], alpha=0.8)
plt.title('Mean Incident Count by Competency Bundle')
plt.ylabel('Average Incident Count')
plt.xlabel('Competency Bundle')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate p-value
plt.text(0.5, max(means) * 0.9, f'Mann-Whitney p={p_value:.4f}', 
         ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()