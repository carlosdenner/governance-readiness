import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import sys

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Define Columns based on debug findings
col_method = '22_dev_method'  # Proxy for Commercial vs Custom
col_eval = '55_independent_eval'

# Filter for rows where both relevant columns are not null (per instructions)
df_analysis = df_eo.dropna(subset=[col_method, col_eval]).copy()

# Define Categories for Procurement Method
# Mapping 'Developed with contracting resources' -> Commercial
# Mapping 'Developed in-house' -> Custom
# Excluding 'Both' to ensure distinct groups for the hypothesis
target_methods = ['Developed with contracting resources.', 'Developed in-house.']
df_analysis = df_analysis[df_analysis[col_method].isin(target_methods)].copy()

df_analysis['Procurement_Type'] = df_analysis[col_method].map({
    'Developed with contracting resources.': 'Commercial',
    'Developed in-house.': 'Custom'
})

# Define Binary Target for Evaluation
# 'Yes...' and 'TRUE' -> 1
# 'Planned', 'Waived', 'Does not apply', etc. -> 0
def is_evaluated(val):
    s = str(val).strip()
    if s.upper() == 'TRUE':
        return 1
    if s.startswith('Yes'):
        return 1
    return 0

df_analysis['Has_Eval'] = df_analysis[col_eval].apply(is_evaluated)

# Generate Contingency Table
contingency = pd.crosstab(df_analysis['Procurement_Type'], df_analysis['Has_Eval'])

# Check if data remains
if contingency.empty:
    print("No valid data intersection found between Development Method and Independent Evaluation.")
    sys.exit(0)

# Ensure columns exist (0 and 1)
if 0 not in contingency.columns: contingency[0] = 0
if 1 not in contingency.columns: contingency[1] = 0
contingency = contingency[[0, 1]]
contingency.columns = ['No Eval', 'Has Eval']

# Statistics
rates = df_analysis.groupby('Procurement_Type')['Has_Eval'].mean()
counts = df_analysis['Procurement_Type'].value_counts()
chi2, p, dof, expected = chi2_contingency(contingency)

# Output
print("--- EXPERIMENT RESULTS: Vendor Opacity Effect ---")
print(f"Data Source: EO 13960 Scored (Filtered for non-null responses)")
print(f"Total valid cases analyzed: {len(df_analysis)}")

print("\nCounts by Procurement Type:")
print(counts)

print("\nContingency Table (Eval Status):")
print(contingency)

print("\nEvaluation Rates:")
for pt in ['Custom', 'Commercial']:
    if pt in rates:
        print(f"  {pt}: {rates[pt]:.2%} ({contingency.loc[pt, 'Has Eval']}/{counts[pt]}) evaluated")

print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4e}")

result_text = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
print(f"\nResult: {result_text}")

# Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(rates.index, rates.values, color=['skyblue', 'orange'])
plt.title(f'Independent Evaluation Rates: Commercial (Vendor) vs Custom (In-House)\n(p={p:.4e})')
plt.ylabel('Proportion with Independent Evaluation')
plt.ylim(0, 1.0)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, 
             f'{height:.1%}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
