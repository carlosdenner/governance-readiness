import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load Data
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for 'eo13960_scored'
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored Subset: {len(eo_df)} rows")

# 3. Clean '22_dev_method' (Procurement Source)
# Map raw values to 'Commercial' or 'In-House'
# We ignore 'Developed with both...' and NaNs for the strict comparison groups
def map_procurement(val):
    s = str(val).strip()
    if s == 'Developed with contracting resources.':
        return 'Commercial'
    elif s == 'Developed in-house.':
        return 'In-House'
    return None

eo_df['procurement_type'] = eo_df['22_dev_method'].apply(map_procurement)

# Filter to only keep the two groups of interest
analysis_df = eo_df.dropna(subset=['procurement_type']).copy()
print(f"Analysis Subset (Commercial vs In-House): {len(analysis_df)} rows")
print(analysis_df['procurement_type'].value_counts())

# 4. Clean '38_code_access'
# Map to Binary: Yes (1) vs No/Missing (0)
# 'Yes' variants include: 'Yes – agency has access...', 'Yes – source code is publicly...', 'Yes', 'YES'
def map_access(val):
    s = str(val).lower()
    if 'yes' in s:
        return 1
    else:
        return 0

analysis_df['has_code_access'] = analysis_df['38_code_access'].apply(map_access)
print("Code Access Distribution:\n", analysis_df['has_code_access'].value_counts())

# 5. Contingency Table
# Rows: Procurement (Commercial, In-House)
# Cols: Code Access (0=No, 1=Yes)
contingency = pd.crosstab(analysis_df['procurement_type'], analysis_df['has_code_access'])
contingency.columns = ['No Access', 'Access']
print("\nContingency Table:\n", contingency)

# 6. Chi-square Test
chi2, p, dof, ex = stats.chi2_contingency(contingency)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# 7. Odds Ratio Calculation
# We want the odds of *losing* code access (No Access) for Commercial vs In-House.
# OR = (Odds No Access | Commercial) / (Odds No Access | In-House)
# Odds No Access = (Count No Access) / (Count Access)

# Extract counts
# Row 'Commercial'
comm_no = contingency.loc['Commercial', 'No Access']
comm_yes = contingency.loc['Commercial', 'Access']

# Row 'In-House'
house_no = contingency.loc['In-House', 'No Access']
house_yes = contingency.loc['In-House', 'Access']

# Add small epsilon if any cell is zero to avoid div by zero (though unlikely here)
if comm_yes == 0 or house_no == 0:
    print("Warning: Zero count detected, adding epsilon.")
    comm_yes += 0.5
    house_no += 0.5
    comm_no += 0.5
    house_yes += 0.5

odds_commercial = comm_no / comm_yes
odds_inhouse = house_no / house_yes

odds_ratio = odds_commercial / odds_inhouse

print(f"Odds of No Access (Commercial): {odds_commercial:.4f}")
print(f"Odds of No Access (In-House): {odds_inhouse:.4f}")
print(f"Odds Ratio (Commercial vs In-House for No Access): {odds_ratio:.4f}")

# 8. Visualization
# Calculate percentages for the plot
rates = analysis_df.groupby('procurement_type')['has_code_access'].mean()
# rates gives % with access. We might want to plot % with access to show the gap.
# If commercial has lower access, its bar will be lower.

plt.figure(figsize=(8, 6))
colors = ['#d9534f', '#5bc0de'] # Red for Commercial, Blue for In-House usually, but let's see order
# rates.index is alphabetical: Commercial, In-House
plt.bar(rates.index, rates.values, color=colors)
plt.ylabel('Proportion with Code Access')
plt.title('Vendor Transparency Gap: Code Access by Procurement Source')
plt.ylim(0, 1.1)

for i, v in enumerate(rates.values):
    plt.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

plt.show()
