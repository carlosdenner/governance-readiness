import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np

# Function to locate file either in current directory or one level up
def get_file_path(filename):
    if os.path.exists(filename):
        return filename
    elif os.path.exists(os.path.join("..", filename)):
        return os.path.join("..", filename)
    else:
        return filename  # Return original to let read_csv fail with clear error if not found

# 1. Load the dataset
file_name = 'step3_tactic_frequency.csv'
file_path = get_file_path(file_name)

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_name}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"Error: Could not find {file_name}")
    exit(1)

# 2. Group the data by 'bundle' (Competency Domain)
# Verify the column name for grouping
if 'bundle' not in df.columns:
    # Fallback based on metadata description if 'bundle' isn't there but 'competency_domain' is
    if 'competency_domain' in df.columns:
        group_col = 'competency_domain'
    else:
        print("Error: Neither 'bundle' nor 'competency_domain' column found.")
        exit(1)
else:
    group_col = 'bundle'

print(f"Grouping by column: {group_col}")
print(f"Unique groups: {df[group_col].unique()}")

# Filter groups
trust_data = df[df[group_col].str.contains('Trust', case=False, na=False)]
integration_data = df[df[group_col].str.contains('Integration', case=False, na=False)]

trust_counts = trust_data['incident_count']
integration_counts = integration_data['incident_count']

# 3. Calculate Descriptive Statistics
t_n = len(trust_counts)
t_mean = trust_counts.mean()
t_median = trust_counts.median()
t_std = trust_counts.std(ddof=1) if t_n > 1 else 0

i_n = len(integration_counts)
i_mean = integration_counts.mean()
i_median = integration_counts.median()
i_std = integration_counts.std(ddof=1) if i_n > 1 else 0

print("\n=== Descriptive Statistics ===")
print(f"Trust Readiness (n={t_n}): Mean={t_mean:.2f}, Median={t_median}, Std={t_std:.2f}")
print(f"Integration Readiness (n={i_n}): Mean={i_mean:.2f}, Median={i_median}, Std={i_std:.2f}")

# 4. Statistical Test (Mann-Whitney U)
# Hypothesis: Integration frequency > Trust frequency
u_stat, p_val = mannwhitneyu(integration_counts, trust_counts, alternative='greater')

print("\n=== Mann-Whitney U Test Results ===")
print(f"Hypothesis: Integration > Trust (frequency)")
print(f"U-statistic: {u_stat}")
print(f"p-value: {p_val:.5f}")

if p_val < 0.05:
    print("Result: Statistically Significant (Reject Null)")
else:
    print("Result: Not Statistically Significant (Fail to Reject Null)")

# 5. Visualization
plt.figure(figsize=(10, 6))
# Create boxplot data
data_to_plot = [trust_counts, integration_counts]
labels = [f'Trust Readiness\n(n={t_n})', f'Integration Readiness\n(n={i_n})']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor="lightblue"), 
            medianprops=dict(color="red"))

plt.title('Incident Frequency Distribution: Trust vs Integration Tactics')
plt.ylabel('Incident Count per Tactic')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()