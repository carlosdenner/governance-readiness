import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

print("Starting Public Accountability Gap analysis (Attempt 2)...")

# 1. Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for Federal AI Inventory data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded EO 13960 data: {len(eo_data)} records")

# 3. Preprocessing with Corrected Logic
col_public = '26_public_service'
col_appeal = '65_appeal_process'

# Define helper function for Public Service classification
def is_public_service(val):
    s = str(val).strip().lower()
    # If it is 'no', empty, or 'nan', it is NOT public facing
    if s in ['no', 'nan', '', 'null']:
        return False
    # Any other descriptive text implies it is a public use case
    return True

# Define helper function for Appeal Process classification
def has_appeal_process(val):
    s = str(val).strip().lower()
    # Only explicit 'yes' counts as having a process
    return s == 'yes'

# Apply mappings
eo_data['is_public'] = eo_data[col_public].apply(is_public_service)
eo_data['has_appeal'] = eo_data[col_appeal].apply(has_appeal_process)

# Print check to ensure we have data in both groups
print("\nDistribution of 'is_public':")
print(eo_data['is_public'].value_counts())
print("\nDistribution of 'has_appeal':")
print(eo_data['has_appeal'].value_counts())

# 4. Analysis
# Contingency Table
contingency = pd.crosstab(eo_data['is_public'], eo_data['has_appeal'])

# Check shape before assigning index/columns to avoid errors if a category is missing
if contingency.shape != (2, 2):
    print("\nWarning: Contingency table is not 2x2. One category may be missing.")
    print(contingency)
else:
    contingency.index = ['Internal/Admin', 'Public-Facing']
    contingency.columns = ['No Appeal Process', 'Has Appeal Process']
    print("\n--- Contingency Table ---")
    print(contingency)

# Calculate Proportions & Stats
results = eo_data.groupby('is_public')['has_appeal'].agg(['count', 'sum', 'mean'])
results.index = ['Internal/Admin', 'Public-Facing'] if len(results) == 2 else results.index
results.columns = ['Total Systems', 'With Appeal', 'Proportion']
print("\n--- Compliance Rates ---")
print(results)

# Statistical Test (Fisher's Exact if sample small, Chi-Square otherwise)
# Given N=1757, Chi-Square is appropriate
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Calculate Odds Ratio
# (Public_Has / Public_Not) / (Internal_Has / Internal_Not)
if contingency.shape == (2, 2):
    a = contingency.iloc[1, 1] # Public, Has Appeal
    b = contingency.iloc[1, 0] # Public, No Appeal
    c = contingency.iloc[0, 1] # Internal, Has Appeal
    d = contingency.iloc[0, 0] # Internal, No Appeal
    
    odds_ratio = (a / b) / (c / d) if (b * c) > 0 else np.nan
    print(f"Odds Ratio: {odds_ratio:.4f}")

# 5. Visualization
plt.figure(figsize=(10, 6))

categories = results.index
proportions = results['Proportion'].values
counts = results['With Appeal'].values
totals = results['Total Systems'].values

colors = ['#6c757d', '#007bff'] if len(categories) == 2 else ['#6c757d']
bars = plt.bar(categories, proportions, color=colors, alpha=0.8)

# Add labels
for bar, count, total, prop in zip(bars, counts, totals, proportions):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{prop:.1%} (n={count}/{total})',
             ha='center', va='bottom')

plt.ylabel('Proportion with Appeal Process')
plt.title('Public Accountability Gap: Availability of Appeal Processes\n(Public-Facing vs. Internal Federal AI Systems)')
plt.ylim(0, max(proportions) * 1.3 if len(proportions) > 0 and max(proportions) > 0 else 0.1)

# Add stats annotation
significance = "Significant" if p < 0.05 else "Not Significant"
plt.annotate(f'p-value: {p:.4e}\n({significance})', 
             xy=(0.5, 0.85), xycoords='axes fraction', 
             ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

plt.tight_layout()
plt.show()
