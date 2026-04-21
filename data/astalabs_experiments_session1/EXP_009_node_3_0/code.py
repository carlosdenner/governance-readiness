import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import sys
import os

# 1. Load the dataset
# Check current directory first, then parent directory to be robust
file_name = 'step2_competency_statements.csv'
if os.path.exists(file_name):
    file_path = file_name
elif os.path.exists(f'../{file_name}'):
    file_path = f'../{file_name}'
else:
    print(f"Error: {file_name} not found.")
    sys.exit(1)

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

# 2. Feature Engineering: Calculate control_count
# applicable_controls is semicolon-separated. Handle NaNs if any.
df['applicable_controls'] = df['applicable_controls'].fillna('')
df['control_count'] = df['applicable_controls'].apply(lambda x: len([c.strip() for c in x.split(';') if c.strip()]))

# Check unique confidence levels
print("\nUnique confidence levels:")
print(df['confidence'].value_counts())

# 3. Grouping: High vs Not High (Medium/Low)
# The hypothesis specifically contrasts High vs others.
df['confidence_group'] = df['confidence'].apply(lambda x: 'High' if str(x).lower() == 'high' else 'Medium/Low')

high_group = df[df['confidence_group'] == 'High']['control_count']
other_group = df[df['confidence_group'] == 'Medium/Low']['control_count']

# Descriptive Statistics
print("\n=== Descriptive Statistics by Confidence Group ===")
group_stats = df.groupby('confidence_group')['control_count'].describe()
print(group_stats)

# 4. Statistical Test (T-test)
# We assume unequal variances (Welch's t-test)
t_stat, p_val = stats.ttest_ind(high_group, other_group, equal_var=False)

print("\n=== Statistical Test Results (Welch's t-test) ===")
print(f"Comparison: High Confidence (n={len(high_group)}) vs Medium/Low Confidence (n={len(other_group)})")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically Significant (Reject Null Hypothesis)")
else:
    print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")

# 5. Visualization
plt.figure(figsize=(10, 6))
sns.violinplot(x='confidence_group', y='control_count', data=df, inner='box', palette='muted')
plt.title('Distribution of Architecture Control Counts by Evidence Confidence')
plt.xlabel('Evidence Confidence')
plt.ylabel('Number of Applicable Controls')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Overlay individual points
sns.stripplot(x='confidence_group', y='control_count', data=df, color='black', alpha=0.5, jitter=True)

plt.show()