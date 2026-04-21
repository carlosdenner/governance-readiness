import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import os

# [debug] Check current directory and file existence
print(f"Current working directory: {os.getcwd()}")
file_name = 'step3_coverage_map.csv'
if os.path.exists(file_name):
    file_path = file_name
elif os.path.exists(f'../{file_name}'):
    file_path = f'../{file_name}'
else:
    print(f"Error: {file_name} not found in current or parent directory.")
    sys.exit(1)

# 1. Load the dataset
try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully from {file_path}.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# 2. Group by 'bundle' and extract 'incident_count'
trust_counts = df[df['bundle'] == 'Trust Readiness']['incident_count']
integration_counts = df[df['bundle'] == 'Integration Readiness']['incident_count']

# 3. Calculate variances and descriptive stats
var_trust = trust_counts.var(ddof=1)
var_integration = integration_counts.var(ddof=1)
mean_trust = trust_counts.mean()
mean_integration = integration_counts.mean()
std_trust = trust_counts.std(ddof=1)
std_integration = integration_counts.std(ddof=1)

print("\n--- Descriptive Statistics ---")
print(f"Trust Readiness:       n={len(trust_counts)}, Mean={mean_trust:.2f}, Variance={var_trust:.2f}, Std Dev={std_trust:.2f}")
print(f"Integration Readiness: n={len(integration_counts)}, Mean={mean_integration:.2f}, Variance={var_integration:.2f}, Std Dev={std_integration:.2f}")

# 4. Perform Levene's Test for equality of variances
# center='median' is robust (Brown-Forsythe)
stat, p_value = stats.levene(trust_counts, integration_counts, center='median')

print("\n--- Levene's Test for Equality of Variances (center='median') ---")
print(f"Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Result: Variances are significantly different (Reject H0).")
else:
    print("Result: Variances are not significantly different (Fail to reject H0).")

# 5. Visualization
plt.figure(figsize=(8, 6))
data_to_plot = [trust_counts, integration_counts]
labels = ['Trust Readiness', 'Integration Readiness']

# Create boxplot
plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', alpha=0.5), 
            medianprops=dict(color='black'))

# Add individual points (jittered x-coordinates for visibility)
import numpy as np
np.random.seed(42)
jitter_trust = np.random.normal(1, 0.04, size=len(trust_counts))
jitter_integration = np.random.normal(2, 0.04, size=len(integration_counts))

plt.scatter(jitter_trust, trust_counts, color='blue', alpha=0.6, label='Trust Data')
plt.scatter(jitter_integration, integration_counts, color='red', alpha=0.6, label='Integration Data')

plt.title('Distribution of Incident Counts by Bundle')
plt.ylabel('Incident Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()
