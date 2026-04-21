import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Check file existence before loading
file_name = 'step2_competency_statements.csv'
file_path = file_name if os.path.exists(file_name) else '../' + file_name

print(f"Loading dataset from {file_path}...")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Could not find {file_name} in current or parent directory.")
    exit(1)

# Verify columns exist
if 'applicable_controls' not in df.columns or 'bundle' not in df.columns:
    print("Error: Required columns not found.")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Calculate architectural density (control_count)
def count_controls(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    # Split by semicolon and strip whitespace to count valid entries
    controls = [c.strip() for c in str(val).split(';') if c.strip()]
    return len(controls)

df['control_count'] = df['applicable_controls'].apply(count_controls)

# Group by bundle
group_integration = df[df['bundle'] == 'Integration Readiness']['control_count']
group_trust = df[df['bundle'] == 'Trust Readiness']['control_count']

# Calculate descriptive statistics
stats_df = df.groupby('bundle')['control_count'].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
print("\n=== Descriptive Statistics (Control Count) ===")
print(stats_df)

# Perform Mann-Whitney U Test (Non-parametric test for difference in distributions)
# We use Mann-Whitney because control counts are discrete and sample sizes are small
u_stat, p_val_mw = stats.mannwhitneyu(group_integration, group_trust, alternative='two-sided')

# Perform Independent T-test (for comparison)
t_stat, p_val_t = stats.ttest_ind(group_integration, group_trust, equal_var=False)

print("\n=== Statistical Test Results ===")
print(f"Mann-Whitney U Test: U={u_stat:.1f}, p-value={p_val_mw:.4f}")
print(f"Welch's T-test:      t={t_stat:.4f}, p-value={p_val_t:.4f}")

interpretation = ""
if p_val_mw < 0.05:
    interpretation = "Result: Statistically significant difference detected."
else:
    interpretation = "Result: No statistically significant difference detected."
print(f"\n{interpretation}")

# Visualization
plt.figure(figsize=(10, 6))
data_to_plot = [group_integration, group_trust]
labels = ['Integration Readiness', 'Trust Readiness']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

plt.title('Distribution of Architecture Control Density by Bundle')
plt.ylabel('Number of Applicable Controls')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add scatter points
for i, data in enumerate(data_to_plot):
    y = data
    x = np.random.normal(i + 1, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.5)

plt.show()