import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the dataset
file_path = 'step3_incident_coding.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    try:
        # Fallback for different directory structure
        df = pd.read_csv('../step3_incident_coding.csv')
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# 2. Preprocess Data
# Handle missing values and normalize strings
df['harm_type'] = df['harm_type'].fillna('').astype(str).str.strip().str.lower()
df['trust_integration_split'] = df['trust_integration_split'].fillna('').astype(str).str.strip().str.lower()

# Filter for specific harm types
target_harms = ['security', 'reliability']
df_subset = df[df['harm_type'].isin(target_harms)].copy()

print(f"Total incidents in analysis: {len(df_subset)}")
print(df_subset['harm_type'].value_counts())

# 3. Create Categorical Variables
# Define 'Both' vs 'Single' (Single includes 'trust-dominant', 'integration-dominant', or missing/other)
df_subset['gap_scope'] = df_subset['trust_integration_split'].apply(lambda x: 'Both Domains' if x == 'both' else 'Single Domain')

# 4. Generate Contingency Table
# explicit crosstab with reindexing to ensure all categories appear
contingency_table = pd.crosstab(df_subset['harm_type'], df_subset['gap_scope'])

# Ensure columns verify 'Single Domain' and 'Both Domains' exist
expected_cols = ['Single Domain', 'Both Domains']
for col in expected_cols:
    if col not in contingency_table.columns:
        contingency_table[col] = 0

# Ensure rows verify 'security' and 'reliability' exist
contingency_table = contingency_table.reindex(target_harms).fillna(0).astype(int)
# Reorder columns for consistency
contingency_table = contingency_table[expected_cols]

print("\nContingency Table (Harm Type vs. Gap Scope):")
print(contingency_table)

# 5. Fisher's Exact Test
# The table is 2x2: [[Security_Single, Security_Both], [Reliability_Single, Reliability_Both]]
odds_ratio, p_value = stats.fisher_exact(contingency_table)

print("\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject Null Hypothesis. Significant association exists.")
else:
    print("Conclusion: Fail to Reject Null Hypothesis. No significant association detected.")

# 6. Visualization
plt.figure(figsize=(8, 6))

# Calculate proportions
prop_table = contingency_table.div(contingency_table.sum(axis=1), axis=0)

# Plot Stacked Bar Chart
ax = prop_table.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], edgecolor='black', width=0.6)

plt.title('Proportion of Gap Scope (Single vs Both) by Harm Type')
plt.xlabel('Harm Type')
plt.ylabel('Proportion of Incidents')
plt.legend(title='Gap Scope', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)

# Annotate bars with values
for c in ax.containers:
    # Filter labels to avoid cluttering if segment is 0
    labels = [f'{v.get_height():.2f}' if v.get_height() > 0 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center')

plt.tight_layout()
plt.show()
