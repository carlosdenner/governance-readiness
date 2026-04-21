import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 records
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"EO 13960 Records: {len(df_eo)}")

# --- Variable Construction ---

# 1. Independent Variable: System Type (Commercial vs Custom)
# Logic: '10_commercial_ai' containing 'None of the above' is treated as Custom/General.
# Any specific commercial category is treated as Commercial.

def classify_type(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    if 'none of the above' in val_str:
        return 'Custom/General'
    else:
        return 'Commercial/COTS'

df_eo['system_type'] = df_eo['10_commercial_ai'].apply(classify_type)

# 2. Dependent Variable: Code Access
# Logic: Explicit 'Yes' vs 'No'. NaNs are excluded from the test.

def classify_access(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    if 'yes' in val_str:
        return 'Yes'
    elif 'no' in val_str:
        return 'No'
    return np.nan

df_eo['code_access'] = df_eo['38_code_access'].apply(classify_access)

# Filter for valid records (excluding NaNs in either variable)
valid_df = df_eo.dropna(subset=['system_type', 'code_access']).copy()

print(f"\nValid Records for Analysis: {len(valid_df)}")
print("\n--- System Type Distribution (Valid Records) ---")
print(valid_df['system_type'].value_counts())

print("\n--- Code Access Distribution (Valid Records) ---")
print(valid_df['code_access'].value_counts())

# --- Statistical Analysis ---

# Contingency Table
contingency = pd.crosstab(valid_df['system_type'], valid_df['code_access'])
# Ensure columns are ordered No, Yes for consistency
if 'No' not in contingency.columns: contingency['No'] = 0
if 'Yes' not in contingency.columns: contingency['Yes'] = 0
contingency = contingency[['No', 'Yes']]

print("\n--- Contingency Table ---")
print(contingency)

# Calculate Rates
comm_row = contingency.loc['Commercial/COTS']
custom_row = contingency.loc['Custom/General']

comm_total = comm_row.sum()
custom_total = custom_row.sum()

comm_rate = comm_row['Yes'] / comm_total if comm_total > 0 else 0
custom_rate = custom_row['Yes'] / custom_total if custom_total > 0 else 0

print(f"\nCommercial Code Access Rate: {comm_rate:.2%} ({comm_row['Yes']}/{comm_total})")
print(f"Custom/General Code Access Rate: {custom_rate:.2%} ({custom_row['Yes']}/{custom_total})")

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpret Result
alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant difference detected.")
else:
    print("Result: No statistically significant difference detected.")

# --- Visualization ---
plt.figure(figsize=(8, 6))
bar_labels = ['Commercial/COTS', 'Custom/General']
bar_values = [comm_rate, custom_rate]
colors = ['#d62728', '#1f77b4']

bars = plt.bar(bar_labels, bar_values, color=colors, alpha=0.8)

plt.ylabel('Proportion with Code Access')
plt.title('Code Access: Commercial vs Custom AI (EO 13960)')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()