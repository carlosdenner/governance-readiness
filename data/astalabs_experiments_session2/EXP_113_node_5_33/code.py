import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Determine Commercial vs Custom using '37_custom_code'
# Rationale: '10_commercial_ai' contains use-case descriptions, not vendor status.
# '37_custom_code' (Yes/No) is a reliable proxy: 'Yes' = Custom/GOTS, 'No' = Commercial/COTS.
valid_code_status = eo_data.dropna(subset=['37_custom_code']).copy()

def map_vendor_type(val):
    val_str = str(val).strip().lower()
    if val_str == 'yes':
        return 'Custom/GOTS'
    elif val_str == 'no':
        return 'Commercial/COTS'
    return None

valid_code_status['Vendor_Type'] = valid_code_status['37_custom_code'].apply(map_vendor_type)
valid_code_status = valid_code_status.dropna(subset=['Vendor_Type'])

# Determine Independent Evaluation Status using '55_independent_eval'
def map_eval_status(val):
    if pd.isna(val):
        return 'No Evaluation'
    val_str = str(val).strip().lower()
    # Strict criteria: Must start with yes or be 'true'
    if val_str.startswith('yes') or val_str == 'true':
        return 'Independent Eval'
    # 'Planned', 'Not applicable', etc. count as No for "have undergone"
    return 'No Evaluation'

valid_code_status['Eval_Status'] = valid_code_status['55_independent_eval'].apply(map_eval_status)

# Generate Contingency Table
contingency = pd.crosstab(valid_code_status['Vendor_Type'], valid_code_status['Eval_Status'])
print("\n--- Contingency Table (Counts) ---")
print(contingency)

# Calculate Proportions
props = pd.crosstab(valid_code_status['Vendor_Type'], valid_code_status['Eval_Status'], normalize='index')
print("\n--- Contingency Table (Proportions) ---")
print(props)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpret results
alpha = 0.05
if p < alpha:
    print("\nResult: Statistically significant relationship found.")
    commercial_eval_rate = props.loc['Commercial/COTS', 'Independent Eval'] if 'Independent Eval' in props.columns else 0
    custom_eval_rate = props.loc['Custom/GOTS', 'Independent Eval'] if 'Independent Eval' in props.columns else 0
    print(f"Commercial Eval Rate: {commercial_eval_rate:.2%}")
    print(f"Custom Eval Rate: {custom_eval_rate:.2%}")
    if commercial_eval_rate < custom_eval_rate:
        print("Hypothesis Supported: Commercial systems are less likely to have independent evaluation.")
    else:
        print("Hypothesis Refuted: Commercial systems are NOT less likely to have independent evaluation.")
else:
    print("\nResult: No statistically significant difference found.")

# Visualization
plt.figure(figsize=(10, 6))
ax = props.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], rot=0)
plt.title('Independent Evaluation Rates: Commercial (No Custom Code) vs Custom (Custom Code)')
plt.ylabel('Proportion')
plt.xlabel('System Type')
plt.legend(title='Evaluation Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Add labels
for c in ax.containers:
    ax.bar_label(c, fmt='%.2f', label_type='center')

plt.show()
