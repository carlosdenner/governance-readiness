import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored subset
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Loaded EO 13960 dataset with {len(eo_data)} records.")

# --- Variable Construction ---

# 1. Define 'is_commercial'
# Logic: If '10_commercial_ai' is NaN or contains 'None of the above', it's Custom (0). Else Commercial (1).
def classify_commercial(val):
    if pd.isna(val):
        return 0
    s = str(val).lower()
    if 'none of the above' in s:
        return 0
    return 1

eo_data['is_commercial'] = eo_data['10_commercial_ai'].apply(classify_commercial)

# 2. Define 'has_ato'
# Logic: Parse column '40_has_ato'. Check if starts with 'yes'.
print("\nUnique values in '40_has_ato' (top 10):")
print(eo_data['40_has_ato'].value_counts().head(10))

def classify_ato(val):
    if pd.isna(val):
        return 0
    s = str(val).lower().strip()
    if s.startswith('yes'):
        return 1
    return 0

eo_data['has_ato'] = eo_data['40_has_ato'].apply(classify_ato)

# --- Analysis ---

# Contingency Table
contingency_table = pd.crosstab(eo_data['is_commercial'], eo_data['has_ato'])
contingency_table.index = ['Custom/Gov', 'Commercial']
contingency_table.columns = ['No ATO', 'Has ATO']

print("\n--- Contingency Table: Commercial Status vs. ATO ---")
print(contingency_table)

# Chi-square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Odds Ratio Calculation
# OR = (ad)/(bc)
# a = Custom, Has ATO
# b = Custom, No ATO
# c = Commercial, Has ATO
# d = Commercial, No ATO
# Note: crosstab structure is:
#             No ATO (0)   Has ATO (1)
# Custom (0)      A            B
# Comm   (1)      C            D
# So OR (Commercial having ATO vs Custom having ATO) = (D/C) / (B/A) = (D*A) / (C*B)

# Extract values
custom_no_ato = contingency_table.loc['Custom/Gov', 'No ATO']
custom_has_ato = contingency_table.loc['Custom/Gov', 'Has ATO']
comm_no_ato = contingency_table.loc['Commercial', 'No ATO']
comm_has_ato = contingency_table.loc['Commercial', 'Has ATO']

# Calculate rates
custom_rate = custom_has_ato / (custom_has_ato + custom_no_ato)
comm_rate = comm_has_ato / (comm_has_ato + comm_no_ato)

print(f"\nATO Rate (Custom/Gov): {custom_rate:.1%} ({custom_has_ato}/{custom_has_ato + custom_no_ato})")
print(f"ATO Rate (Commercial): {comm_rate:.1%} ({comm_has_ato}/{comm_has_ato + comm_no_ato})")

try:
    odds_ratio = (comm_has_ato * custom_no_ato) / (comm_no_ato * custom_has_ato)
    print(f"\nOdds Ratio (Commercial vs Custom for having ATO): {odds_ratio:.4f}")
    
    # Inverse OR for interpretation if Custom is higher
    if odds_ratio < 1:
        inv_or = 1 / odds_ratio
        print(f"Interpretation: Custom systems are {inv_or:.2f} times more likely to have an ATO than Commercial systems.")
    else:
        print(f"Interpretation: Commercial systems are {odds_ratio:.2f} times more likely to have an ATO than Custom systems.")
except ZeroDivisionError:
    print("\nCannot calculate Odds Ratio due to zero division.")

# Visualization
plt.figure(figsize=(8, 6))
rates = [custom_rate, comm_rate]
labels = ['Custom/Gov', 'Commercial']
colors = ['#1f77b4', '#ff7f0e']

bars = plt.bar(labels, rates, color=colors)
plt.ylabel('ATO Compliance Rate')
plt.title('ATO Compliance: Custom vs. Commercial AI')
plt.ylim(0, 1.0)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.1%}", ha='center', va='bottom')

plt.show()
