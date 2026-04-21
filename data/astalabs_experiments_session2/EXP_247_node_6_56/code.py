import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# 1. Define Source Type using '37_custom_code'
# 'Yes' -> Custom, 'No' -> Commercial
def classify_source(val):
    s = str(val).strip().lower()
    if s == 'yes':
        return 'Custom'
    elif s == 'no':
        return 'Commercial'
    return None

eo_data['source_type'] = eo_data['37_custom_code'].apply(classify_source)

# 2. Define Code Access using '38_code_access'
# Map values starting with 'Yes' to 1, 'No' to 0
def classify_access(val):
    s = str(val).strip().lower()
    if s.startswith('yes'):
        return 1
    elif s.startswith('no'):
        return 0
    return None

eo_data['has_code_access'] = eo_data['38_code_access'].apply(classify_access)

# Drop rows with missing values in the relevant columns
analysis_df = eo_data.dropna(subset=['source_type', 'has_code_access']).copy()

print(f"Rows for analysis: {len(analysis_df)}")
print("Source distribution:\n", analysis_df['source_type'].value_counts())
print("Code Access distribution:\n", analysis_df['has_code_access'].value_counts())

# 3. Contingency Table
contingency_table = pd.crosstab(analysis_df['source_type'], analysis_df['has_code_access'])
contingency_table.columns = ['No Access', 'Has Access']
print("\nContingency Table (Source Type x Code Access):\n", contingency_table)

# 4. Statistical Test
# Using Chi-Square Test of Independence
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p_value:.5e}")

# 5. Calculate Percentages for Plotting
summary = analysis_df.groupby('source_type')['has_code_access'].agg(['count', 'mean'])
summary['percent_access'] = summary['mean'] * 100
print("\nAccess Rates:\n", summary[['count', 'percent_access']])

# 6. Visualization
plt.figure(figsize=(8, 6))
colors = ['#d62728', '#1f77b4'] # Red for Commercial, Blue for Custom
bars = plt.bar(summary.index, summary['percent_access'], color=colors)
plt.title(f'Code Access Availability: Commercial vs. Custom AI\n(p={p_value:.2e})')
plt.xlabel('Source Type')
plt.ylabel('Percentage with Code Access (%)')
plt.ylim(0, 100)

# Add labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
