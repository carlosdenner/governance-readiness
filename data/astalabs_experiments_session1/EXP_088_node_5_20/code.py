import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Define the file name
filename = 'step3_coverage_map.csv'

# Try to locate the file
file_path = None
possible_paths = [filename, os.path.join('..', filename)]

for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    print(f"Error: Could not find {filename} in current or parent directory.")
    print(f"Current working directory: {os.getcwd()}")
    sys.exit(1)

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Function to count distinct harms
def count_harms(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    # Split by semicolon
    parts = str(val).split(';')
    # Filter empty strings
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)

# Apply counting
df['distinct_harm_count'] = df['primary_harm_types'].apply(count_harms)

# Filter for relevant bundles
valid_bundles = ['Trust Readiness', 'Integration Readiness']
df_filtered = df[df['bundle'].isin(valid_bundles)].copy()

# Group data
trust_data = df_filtered[df_filtered['bundle'] == 'Trust Readiness']['distinct_harm_count']
integration_data = df_filtered[df_filtered['bundle'] == 'Integration Readiness']['distinct_harm_count']

# Descriptive Statistics
print("\n=== Descriptive Statistics: Distinct Harm Count by Bundle ===")
print(f"Trust Readiness (n={len(trust_data)}): Mean={trust_data.mean():.2f}, Std={trust_data.std():.2f}")
print(f"Integration Readiness (n={len(integration_data)}): Mean={integration_data.mean():.2f}, Std={integration_data.std():.2f}")

# T-Test (Welch's)
t_stat, p_val = stats.ttest_ind(trust_data, integration_data, equal_var=False)
print("\n=== Welch's T-Test Results ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# Visualization
plt.figure(figsize=(8, 6))

# Jitter for strip plot
jitter_trust = np.random.normal(0, 0.05, size=len(trust_data))
jitter_integ = np.random.normal(0, 0.05, size=len(integration_data))

plt.scatter(np.zeros_like(trust_data) + jitter_trust, trust_data, 
            label='Trust Readiness', alpha=0.7, s=100, color='skyblue', edgecolors='black')
plt.scatter(np.ones_like(integration_data) + jitter_integ, integration_data, 
            label='Integration Readiness', alpha=0.7, s=100, color='salmon', edgecolors='black')

# Plot Means
plt.plot([-0.2, 0.2], [trust_data.mean(), trust_data.mean()], color='blue', lw=3, linestyle='--', label='Mean Trust')
plt.plot([0.8, 1.2], [integration_data.mean(), integration_data.mean()], color='red', lw=3, linestyle='--', label='Mean Integration')

plt.xticks([0, 1], ['Trust Readiness', 'Integration Readiness'])
plt.ylabel('Count of Distinct Harm Types')
plt.title('Risk Scope: Distinct Harms per Sub-Competency')
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.6)

plt.show()
