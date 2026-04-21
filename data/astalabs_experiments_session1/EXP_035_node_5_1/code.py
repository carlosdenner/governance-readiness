import subprocess
import sys
import os

# Function to install packages if not found
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Import libraries with fallback installation
try:
    import pandas as pd
except ImportError:
    install('pandas')
    import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    install('matplotlib')
    import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    install('scipy')
    from scipy.stats import mannwhitneyu

# Load dataset
file_name = 'step3_coverage_map.csv'
file_path = f'../{file_name}'

# Check file existence and load
if not os.path.exists(file_path):
    if os.path.exists(file_name):
        file_path = file_name
    else:
        print(f"Error: {file_name} not found in parent or current directory.")
        sys.exit(1)

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Filter data by bundle
trust_data = df[df['bundle'] == 'Trust Readiness']
integration_data = df[df['bundle'] == 'Integration Readiness']

trust_counts = trust_data['incident_count']
integration_counts = integration_data['incident_count']

# Descriptive Statistics
print("\n--- Descriptive Statistics ---")
print(f"Trust Readiness (n={len(trust_counts)}):")
print(trust_counts.describe())
print(f"\nIntegration Readiness (n={len(integration_counts)}):")
print(integration_counts.describe())

# Mann-Whitney U Test
# Hypothesis: Integration > Trust (Alternative = 'greater')
stat, p_val = mannwhitneyu(integration_counts, trust_counts, alternative='greater')

print("\n--- Mann-Whitney U Test Results ---")
print(f"Hypothesis: Integration Readiness incident counts > Trust Readiness incident counts")
print(f"U-statistic: {stat}")
print(f"P-value: {p_val:.5f}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically significant difference detected.")
else:
    print("Result: No statistically significant difference detected.")

# Boxplot Visualization
plt.figure(figsize=(10, 6))
data_to_plot = [trust_counts, integration_counts]
labels = ['Trust Readiness', 'Integration Readiness']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
plt.title('Distribution of Incident Counts by Competency Bundle')
plt.ylabel('Number of Mapped Incidents')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add jittered points to show individual data points since N is small
import numpy as np
for i, data in enumerate(data_to_plot):
    y = data
    x = np.random.normal(1 + i, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.6)

plt.show()