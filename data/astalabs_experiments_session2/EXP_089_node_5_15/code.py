import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load the dataset
filename = 'astalabs_discovery_all_data.csv'
df = pd.read_csv(filename, low_memory=False)

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# 2. Clean '40_has_ato' (Authority to Operate)
# Positive values indicating an ATO or equivalent approved status
ato_positive = ['Yes', 'Operated in an approved enclave']

# Function to clean ATO column
def clean_ato(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip()
    if val_str in ato_positive:
        return 1
    return 0

eo_data['has_ato_binary'] = eo_data['40_has_ato'].apply(clean_ato)

# 3. Clean '52_impact_assessment'
# Positive values indicating a documented assessment
impact_positive = ['Yes', 'YES']

# Function to clean Impact Assessment column
def clean_impact(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip()
    if val_str in impact_positive:
        return 1
    return 0

eo_data['has_impact_binary'] = eo_data['52_impact_assessment'].apply(clean_impact)

# 4. Create Contingency Table
contingency_table = pd.crosstab(eo_data['has_ato_binary'], eo_data['has_impact_binary'])
contingency_table.index = ['No ATO', 'Has ATO']
contingency_table.columns = ['No Impact Assessment', 'Has Impact Assessment']

print("--- Contingency Table ---")
print(contingency_table)
print("\n")

# 5. Perform Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# 6. Calculate Odds Ratio
# OR = (a*d) / (b*c) where a=HasATO_HasImpact, b=HasATO_NoImpact, c=NoATO_HasImpact, d=NoATO_NoImpact
# But crosstab order is [0,0], [0,1], [1,0], [1,1]
# NoATO_NoImpact (0,0), NoATO_HasImpact (0,1)
# HasATO_NoImpact (1,0), HasATO_HasImpact (1,1)

n00 = contingency_table.iloc[0, 0] # No ATO, No Impact
n01 = contingency_table.iloc[0, 1] # No ATO, Has Impact
n10 = contingency_table.iloc[1, 0] # Has ATO, No Impact
n11 = contingency_table.iloc[1, 1] # Has ATO, Has Impact

# Use Haldane-Anscombe correction if any cell is 0 (add 0.5), though unlikely here with n=1757
if (n00==0 or n01==0 or n10==0 or n11==0):
    odds_ratio = ((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5))
else:
    odds_ratio = (n11 * n00) / (n10 * n01)

print(f"--- Chi-Square Results ---")
print(f"Chi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Odds Ratio: {odds_ratio:.4f}")

# Calculate percentages for plotting
ato_compliance_rate = n11 / (n11 + n10) * 100
no_ato_compliance_rate = n01 / (n01 + n00) * 100

print(f"\nCompliance Rate (Has ATO): {ato_compliance_rate:.2f}%")
print(f"Compliance Rate (No ATO): {no_ato_compliance_rate:.2f}%")

# 7. Visualization
labels = ['No ATO', 'Has ATO']
rates = [no_ato_compliance_rate, ato_compliance_rate]

plt.figure(figsize=(8, 6))
plt.bar(labels, rates, color=['#e74c3c', '#2ecc71'], alpha=0.8)
plt.ylabel('Percentage with Documented Impact Assessment')
plt.title('Impact Assessment Compliance by ATO Status')
plt.ylim(0, max(rates) * 1.2)

for i, v in enumerate(rates):
    plt.text(i, v + 0.2, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
