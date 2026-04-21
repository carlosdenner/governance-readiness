import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# [debug]
print("Starting experiment: Defense vs Civilian Transparency...")

# 1. Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        print("Dataset loaded from current directory.")
    except FileNotFoundError:
        print("Error: Dataset not found in ../ or ./")
        exit(1)

# 2. Filter for eo13960_scored
# source_table column identifies the subset
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset 'eo13960_scored' size: {len(subset)}")

# 3. Categorize Agencies
# Inspect unique agencies to ensure robust mapping
# unique_agencies = subset['3_agency'].unique()
# print("Unique Agencies:", unique_agencies)

def categorize_agency(agency_name):
    if pd.isna(agency_name):
        return 'Unknown'
    agency_lower = str(agency_name).lower()
    defense_keywords = ['defense', 'homeland', 'justice', 'state', 'intelligence', 'army', 'navy', 'air force', 'dod', 'dhs', 'doj']
    if any(keyword in agency_lower for keyword in defense_keywords):
        return 'Defense/Security'
    else:
        return 'Civilian'

subset['Agency_Type'] = subset['3_agency'].apply(categorize_agency)
subset = subset[subset['Agency_Type'] != 'Unknown']

# 4. Create binary Has_Appeal_Process
# Inspect column 65_appeal_process
print("Unique values in '65_appeal_process':", subset['65_appeal_process'].unique())

def check_appeal(val):
    if pd.isna(val):
        return False
    # Check for affirmative 'Yes' or similar variants
    val_str = str(val).lower().strip()
    return val_str == 'yes' or val_str == 'true'

subset['Has_Appeal'] = subset['65_appeal_process'].apply(check_appeal)

# 5. Create Contingency Table
contingency_table = pd.crosstab(subset['Agency_Type'], subset['Has_Appeal'])
contingency_table.columns = ['No Appeal Process', 'Has Appeal Process']
print("\nContingency Table:")
print(contingency_table)

# 6. Statistical Test (Chi-Square)
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Calculate percentages for clearer interpretation
summary = contingency_table.copy()
summary['Total'] = summary.sum(axis=1)
summary['% with Appeal'] = (summary['Has Appeal Process'] / summary['Total']) * 100
print("\nSummary Statistics:")
print(summary)

# 7. Visualization
# Plotting % with Appeal Process
plt.figure(figsize=(8, 6))
agency_types = summary.index
percentages = summary['% with Appeal']

bars = plt.bar(agency_types, percentages, color=['skyblue', 'salmon'])
plt.title('Percentage of AI Systems with Appeal Processes by Agency Type')
plt.xlabel('Agency Category')
plt.ylabel('Percentage Reporting Appeal Process (%)')
plt.ylim(0, max(percentages) * 1.2 if max(percentages) > 0 else 10)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{height:.1f}%', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nResult: The difference in appeal process availability is statistically significant.")
    if summary.loc['Defense/Security', '% with Appeal'] < summary.loc['Civilian', '% with Appeal']:
        print("Hypothesis Supported: Defense/Security agencies report significantly fewer appeal processes.")
    else:
        print("Hypothesis Refuted: Defense/Security agencies report significantly MORE appeal processes.")
else:
    print("\nResult: No statistically significant difference found between agency types.")
