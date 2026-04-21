import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# Normalize Agency Abbreviation
eo_df['3_abr'] = eo_df['3_abr'].astype(str).str.upper().str.strip()

# Define Agency Types
defense_codes = ['DOD', 'DHS', 'DOJ']
eo_df['Agency_Type'] = eo_df['3_abr'].apply(lambda x: 'Defense/Security' if x in defense_codes else 'Civilian')

# Define logic to parse '59_ai_notice'
def parse_notice(val):
    s = str(val).lower().strip()
    # Negative indicators
    if s == 'nan' or s == '': return 0
    if 'none of the above' in s: return 0
    if 'n/a' in s: return 0
    if 'waived' in s: return 0
    if 'not safety' in s: return 0
    
    # Affirmative indicators (if not caught by above)
    # The previous output showed values like 'Online', 'In-person', 'Email', 'Telephone', 'Other'
    # Since we filtered out negatives, we assume the rest are affirmative forms of notice.
    return 1

# Apply parsing
eo_df['Has_Notice'] = eo_df['59_ai_notice'].apply(parse_notice)

# Calculate Rates
rates = eo_df.groupby('Agency_Type')['Has_Notice'].agg(['count', 'sum', 'mean'])
rates.columns = ['Total Systems', 'Systems with Notice', 'Notice Rate']

print("--- Transparency Rates by Agency Type ---")
print(rates)
print("\n")

# Contingency Table for Chi-Square
contingency = pd.crosstab(eo_df['Agency_Type'], eo_df['Has_Notice'])
print("--- Contingency Table (0=No Notice, 1=Notice) ---")
print(contingency)
print("\n")

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print("--- Chi-square Test Results ---")
print(f"Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpretation
alpha = 0.05
is_significant = p < alpha
print(f"\nStatistically Significant: {is_significant}")
if is_significant:
    def_rate = rates.loc['Defense/Security', 'Notice Rate']
    civ_rate = rates.loc['Civilian', 'Notice Rate']
    if def_rate < civ_rate:
        print("Direction: Defense agencies have significantly LOWER transparency.")
    else:
        print("Direction: Defense agencies have significantly HIGHER transparency.")
else:
    print("No significant difference in transparency rates.")

# Visualization
plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#d62728'] # Blue for Civilian, Red for Defense
ax = rates['Notice Rate'].plot(kind='bar', color=colors, rot=0)
plt.title('Public AI Notice Compliance: Civilian vs Defense')
plt.ylabel('Proportion of Systems with Public Notice')
plt.ylim(0, 1.0)

# Add value labels
for i, v in enumerate(rates['Notice Rate']):
    ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()