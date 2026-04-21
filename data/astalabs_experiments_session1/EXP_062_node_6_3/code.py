import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
import os

# Load the dataset
filename = 'step2_crosswalk_matrix.csv'

# Fallback to check parent directory if not found locally, just in case
if not os.path.exists(filename):
    filename = '../' + filename

df = pd.read_csv(filename)

# Define the columns of interest
col_ops = 'GenAIOps / MLOps Lifecycle Governance'
col_eval = 'Evaluation & Monitoring Infrastructure'

# Preprocess: Convert 'X' to 1, others to 0
df['Ops_Binary'] = df[col_ops].apply(lambda x: 1 if str(x).strip() == 'X' else 0)
df['Eval_Binary'] = df[col_eval].apply(lambda x: 1 if str(x).strip() == 'X' else 0)

# Create Contingency Table
contingency_table = pd.crosstab(df['Ops_Binary'], df['Eval_Binary'])

# Rename index/columns for clarity (0=No, 1=Yes)
contingency_table.index = ['No Ops Gov', 'Has Ops Gov']
contingency_table.columns = ['No Eval Infra', 'Has Eval Infra']

# Calculate Statistics
# 1. Chi-square test (correction=False for raw chi2)
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table, correction=False)

# 2. Fisher's Exact Test
odds_ratio, p_value_fisher = stats.fisher_exact(contingency_table)

# 3. Phi Coefficient
phi_coeff = matthews_corrcoef(df['Ops_Binary'], df['Eval_Binary'])

print("=== Co-occurrence Analysis ===")
print(f"File Loaded: {filename}")
print(f"Dataset shape: {df.shape}")
print(f"\nContingency Table:\n{contingency_table}")
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value (Chi-square): {p_value:.4f}")
print(f"P-value (Fisher's Exact): {p_value_fisher:.4f}")
print(f"Phi Coefficient: {phi_coeff:.4f}")

# Interpretation
alpha = 0.05
if p_value_fisher < alpha:
    print("\nResult: Statistically significant co-occurrence detected.")
else:
    print("\nResult: No statistically significant co-occurrence detected (Null hypothesis not rejected).")

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Co-occurrence of GenAIOps and Eval Infrastructure')
plt.xlabel('Evaluation & Monitoring Infrastructure')
plt.ylabel('GenAIOps / MLOps Lifecycle Governance')
plt.tight_layout()
plt.show()