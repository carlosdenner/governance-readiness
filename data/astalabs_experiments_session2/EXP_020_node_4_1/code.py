import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Total EO13960 records: {len(eo_data)}")

# Target columns
date_col = '20_date_implemented'
impact_col = '52_impact_assessment'

# Check if columns exist
if date_col not in eo_data.columns or impact_col not in eo_data.columns:
    print(f"Error: Missing required columns. Available columns: {eo_data.columns.tolist()}")
    exit()

# Parse dates
# The column might contain various formats, force coercion
eo_data['year'] = pd.to_datetime(eo_data[date_col], errors='coerce').dt.year

# Filter for valid years in the range of interest (2020-2023)
eo_data = eo_data[eo_data['year'].isin([2020, 2021, 2022, 2023])]
print(f"Records in 2020-2023 with valid dates: {len(eo_data)}")

# Define periods
def get_period(year):
    if year in [2020, 2021]:
        return 'Crisis (2020-2021)'
    elif year in [2022, 2023]:
        return 'Post-Crisis (2022-2023)'
    return None

eo_data['period'] = eo_data['year'].apply(get_period)

# Clean Impact Assessment column
# Normalize to boolean: True if 'Yes', False otherwise
# Let's inspect unique values first to be safe
print(f"Unique values in {impact_col}: {eo_data[impact_col].unique()}")

# normalizing
valid_yes = ['yes', 'true', '1', 'y']
eo_data['has_impact_assessment'] = eo_data[impact_col].astype(str).str.lower().isin(valid_yes)

# Group analysis
summary = eo_data.groupby('period')['has_impact_assessment'].agg(['count', 'sum', 'mean']).rename(columns={'count': 'Total', 'sum': 'Compliant', 'mean': 'Rate'})
print("\nSummary Statistics by Period:")
print(summary)

# Chi-Square Test
# Contingency table: [[Crisis_Compliant, Crisis_NonCompliant], [Post_Compliant, Post_NonCompliant]]
crisis_stats = summary.loc['Crisis (2020-2021)']
post_stats = summary.loc['Post-Crisis (2022-2023)']

contingency_table = [
    [int(crisis_stats['Compliant']), int(crisis_stats['Total'] - crisis_stats['Compliant'])],
    [int(post_stats['Compliant']), int(post_stats['Total'] - post_stats['Compliant'])]
]

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n--- Statistical Test Results ---")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4f}")

if p < 0.05:
    print("Result: Significant difference found.")
else:
    print("Result: No significant difference found.")

# Visualization
plt.figure(figsize=(10, 6))

# 1. Bar chart for Period comparison
periods = summary.index
rates = summary['Rate'] * 100

colors = ['#ff9999', '#66b3ff']
bars = plt.bar(periods, rates, color=colors, edgecolor='black', alpha=0.7)

plt.title('Impact Assessment Compliance Rate: Crisis vs Post-Crisis', fontsize=14)
plt.ylabel('Compliance Rate (%)', fontsize=12)
plt.ylim(0, 100)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# 2. Time Series Plot by Year
yearly_summary = eo_data.groupby('year')['has_impact_assessment'].mean() * 100
plt.figure(figsize=(10, 6))
plt.plot(yearly_summary.index, yearly_summary.values, marker='o', linestyle='-', color='purple', linewidth=2)
plt.title('Impact Assessment Compliance Rate over Time (2020-2023)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Compliance Rate (%)', fontsize=12)
plt.ylim(0, 100)
plt.xticks([2020, 2021, 2022, 2023])
plt.grid(True, linestyle='--', alpha=0.5)
for x, y in zip(yearly_summary.index, yearly_summary.values):
    plt.text(x, y + 2, f'{y:.1f}%', ha='center', fontweight='bold')
plt.show()
