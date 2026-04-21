import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def load_dataset(filename):
    # Check current directory
    if os.path.exists(filename):
        return pd.read_csv(filename)
    # Check parent directory
    elif os.path.exists(os.path.join('..', filename)):
        return pd.read_csv(os.path.join('..', filename))
    else:
        raise FileNotFoundError(f"Could not find {filename} in current or parent directory.")

# 1. Load dataset
filename = 'step2_crosswalk_matrix.csv'
try:
    df = load_dataset(filename)
    print(f"Successfully loaded {filename}")
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

# 2. Filter for relevant sources
# We want to compare OWASP Top 10 LLM (Security focus) vs NIST AI RMF 1.0 (Risk focus)
target_sources = ['OWASP Top 10 LLM', 'NIST AI RMF 1.0']

# Filter and create a copy to avoid SettingWithCopyWarning
filtered_df = df[df['source'].isin(target_sources)].copy()

print(f"\nFiltered Data Shape: {filtered_df.shape}")
print("Source Counts:\n", filtered_df['source'].value_counts())
print("Bundle Counts:\n", filtered_df['bundle'].value_counts())

# 3. Create Contingency Table
# Rows: Source, Columns: Bundle
contingency_table = pd.crosstab(filtered_df['source'], filtered_df['bundle'])
print("\nContingency Table (Observed):\n", contingency_table)

# 4. Perform Chi-Square Test of Independence
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n=== Chi-Square Test Results ===")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")

print("\nExpected Frequencies:\n", pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

# Interpretation
alpha = 0.05
print("\n=== Interpretation ===")
if p < alpha:
    print(f"P-value ({p:.4f}) < {alpha}: Reject Null Hypothesis.")
    print("There is a statistically significant association between the Framework Source and the Competency Bundle.")
    print("This supports the hypothesis of framework bias (Security vs Risk).")
else:
    print(f"P-value ({p:.4f}) >= {alpha}: Fail to Reject Null Hypothesis.")
    print("There is no statistically significant association found.")

# 5. Visualization
plt.figure(figsize=(10, 6))

# Calculate proportions for stacked bar chart
contingency_props = contingency_table.div(contingency_table.sum(1), axis=0)

# Plot
ax = contingency_props.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'], alpha=0.9)

plt.title('Competency Bundle Distribution by Framework Source')
plt.xlabel('Framework Source')
plt.ylabel('Proportion of Requirements')
plt.legend(title='Competency Bundle', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Annotate bars with counts and percentages
for i, (idx, row) in enumerate(contingency_props.iterrows()):
    cum_height = 0
    for col in contingency_props.columns:
        height = row[col]
        count = contingency_table.loc[idx, col]
        if height > 0:
            plt.text(i, cum_height + height/2, f"{count}\n({height:.1%})", 
                     ha='center', va='center', color='white', fontweight='bold')
            cum_height += height

plt.tight_layout()
plt.show()