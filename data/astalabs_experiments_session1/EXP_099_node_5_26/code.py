import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Handle file loading with path check
filename = 'step2_crosswalk_matrix.csv'
file_path = f"../{filename}" if os.path.exists(f"../{filename}") else filename

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# 2. Create standardized function variable
# 'GOVERN' and 'GOVERNANCE' -> 'Strategic'
# 'MAP', 'MEASURE', 'MANAGE' -> 'Operational'
def classify_function(func):
    if pd.isna(func):
        return None
    func = str(func).upper().strip()
    if func in ['GOVERN', 'GOVERNANCE']:
        return 'Strategic'
    elif func in ['MAP', 'MEASURE', 'MANAGE']:
        return 'Operational'
    else:
        return 'Other'

df['function_category'] = df['function'].apply(classify_function)

# Filter for only Strategic and Operational to test the specific hypothesis
analysis_df = df[df['function_category'].isin(['Strategic', 'Operational'])].copy()

print(f"\nData filtered for analysis (n={len(analysis_df)}):")
print(analysis_df['function_category'].value_counts())

# 3. Create Contingency Table
contingency_table = pd.crosstab(analysis_df['function_category'], analysis_df['bundle'])

print("\nContingency Table (Observed):")
print(contingency_table)

# 4. Perform Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n=== Chi-Square Test Results ===")
print(f"Chi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nResult: Statistically Significant. The variables are dependent.")
else:
    print("\nResult: Not Statistically Significant. The variables are independent.")

# 5. Visualization
plt.figure(figsize=(10, 6))
# Calculate proportions for stacked bar
cross_tab_prop = pd.crosstab(index=analysis_df['function_category'],
                             columns=analysis_df['bundle'],
                             normalize="index")

cross_tab_prop.plot(kind='bar', stacked=True, color=['#4c72b0', '#55a868'], figsize=(10, 6))

plt.title('Proportion of Readiness Bundles by Framework Function Category')
plt.xlabel('Function Category')
plt.ylabel('Proportion')
plt.legend(title='Competency Bundle', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()