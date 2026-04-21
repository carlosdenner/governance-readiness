import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Locate the dataset
filename = 'step3_tactic_frequency.csv'
paths_to_check = [filename, f"../{filename}"]
file_path = None

for path in paths_to_check:
    if os.path.exists(path):
        file_path = path
        break

if not file_path:
    raise FileNotFoundError(f"Could not find {filename} in current or parent directory.")

# 1. Load the dataset
df = pd.read_csv(file_path)
print(f"Successfully loaded {filename}")
print(df.head())

# 2. Group data by 'bundle'
trust_counts = df[df['bundle'] == 'Trust Readiness']['incident_count']
integration_counts = df[df['bundle'] == 'Integration Readiness']['incident_count']

# 3. Descriptive Statistics
mean_trust = trust_counts.mean()
mean_integration = integration_counts.mean()
std_trust = trust_counts.std()
std_integration = integration_counts.std()

print("\n=== Descriptive Statistics ===")
print(f"Trust Readiness:       n={len(trust_counts)}, Mean={mean_trust:.2f}, Std Dev={std_trust:.2f}")
print(f"Integration Readiness: n={len(integration_counts)}, Mean={mean_integration:.2f}, Std Dev={std_integration:.2f}")

# 4. Perform T-test (Welch's t-test for unequal variances)
t_stat, p_val = stats.ttest_ind(integration_counts, trust_counts, equal_var=False)

print("\n=== Statistical Test Results (Welch's t-test) ===")
print(f"Hypothesis: Integration > Trust")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically Significant Difference")
else:
    print("Result: No Statistically Significant Difference")

# 5. Visualization
plt.figure(figsize=(8, 6))
plt.boxplot([trust_counts, integration_counts], labels=['Trust Readiness', 'Integration Readiness'])
plt.title('Adversarial Tactic Frequency by Competency Bundle')
plt.ylabel('Incident Frequency (Count)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()