import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
import os

# [debug] Check file existence
file_name = 'step3_incident_coding.csv'
path = f"../{file_name}"
if not os.path.exists(path):
    path = file_name # Fallback to current directory

print(f"Loading dataset from: {path}")

try:
    df = pd.read_csv(path)
except FileNotFoundError:
    print(f"Error: Could not find {file_name} in ../ or current directory.")
    exit(1)

# Define categorization logic
security_harms = ['security', 'supply_chain']
# Any harm type not in security_harms is considered Non-Security for this hypothesis

def categorize_harm(harm_type):
    if harm_type in security_harms:
        return 'Security'
    else:
        return 'Non-Security'

# Apply categorization
if 'harm_type' in df.columns:
    df['Harm_Category'] = df['harm_type'].apply(categorize_harm)
else:
    print("Column 'harm_type' not found.")
    exit(1)

# Ensure technique_count exists
if 'technique_count' not in df.columns:
    # Attempt to calculate if missing, though metadata says it exists
    if 'techniques_used' in df.columns:
        df['technique_count'] = df['techniques_used'].astype(str).apply(lambda x: len(x.split(';')) if x.lower() != 'nan' else 0)
    else:
        print("Column 'technique_count' or 'techniques_used' not found.")
        exit(1)

# Separate groups
security_group = df[df['Harm_Category'] == 'Security']['technique_count']
non_security_group = df[df['Harm_Category'] == 'Non-Security']['technique_count']

# Descriptive Statistics
print("\n=== Descriptive Statistics (Technique Count) ===")
print(f"Security Group (n={len(security_group)}):")
print(security_group.describe())
print(f"\nNon-Security Group (n={len(non_security_group)}):")
print(non_security_group.describe())

# Statistical Test (Mann-Whitney U is preferred due to likely non-normal distribution and unequal sample sizes)
stat, p_value = mannwhitneyu(security_group, non_security_group, alternative='two-sided')

print("\n=== Statistical Test Results (Mann-Whitney U) ===")
print(f"U-statistic: {stat}")
print(f"P-value: {p_value:.5f}")

alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant difference found.")
else:
    print("Result: No statistically significant difference found.")

# Visualization
plt.figure(figsize=(8, 6))
data_to_plot = [security_group, non_security_group]
labels = [f'Security\n(n={len(security_group)})', f'Non-Security\n(n={len(non_security_group)})']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
plt.title('Distribution of Technique Counts by Harm Category')
plt.ylabel('Number of Techniques Used')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
