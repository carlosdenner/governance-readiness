import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# [debug] List files in parent directory to ensure path is correct
# print(os.listdir('../'))

# 1. Load the dataset
file_path = '../step2_crosswalk_matrix.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    # Fallback if running in same dir
    df = pd.read_csv('step2_crosswalk_matrix.csv')

# 2. Group data by 'source' and 'bundle'
# Clean up source names if necessary, but inspection suggests they are distinct categories
# The prompt mentions: NIST AI RMF 1.0, NIST GenAI Profile, EU AI Act, OWASP Top 10 LLM

# Create Contingency Table
contingency_table = pd.crosstab(df['source'], df['bundle'])

print("=== Contingency Table (Count) ===")
print(contingency_table)
print("\n")

# 3. Perform Chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("=== Chi-square Test Results ===")
print(f"Chi-square Statistic: {chi2:.4f}")
print(f"p-value: {p:.4e}")
print(f"Degrees of Freedom: {dof}")

alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant (Reject H0). Source and Bundle are associated.")
else:
    print("Result: Not Statistically Significant (Fail to reject H0). No evidence of association.")
print("\n")

# 4. Calculate percentages
# Normalize by row (Source) to see the split per framework
contingency_pct = pd.crosstab(df['source'], df['bundle'], normalize='index') * 100
print("=== Distribution by Source (%) ===")
print(contingency_pct.round(2))
print("\n")

# 5. Generate Stacked Bar Chart
# Set plot style
plt.style.use('ggplot')

ax = contingency_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')

plt.title('Competency Bundle Distribution by Normative Framework')
plt.xlabel('Normative Framework (Source)')
plt.ylabel('Percentage of Requirements (%)')
plt.legend(title='Competency Bundle', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')

# Annotate bars
for c in ax.containers:
    ax.bar_label(c, fmt='%.0f%%', label_type='center', color='white', weight='bold')

plt.tight_layout()
plt.show()
