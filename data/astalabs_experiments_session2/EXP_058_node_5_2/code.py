import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import os

# --- 1. Load Data ---
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(f'../{filename}'):
    filepath = f'../{filename}'
else:
    filepath = filename

print(f"Loading dataset from: {filepath}")
try:
    df = pd.read_csv(filepath, low_memory=False)
except Exception as e:
    print(f"Failed to load csv: {e}")
    sys.exit(1)

# Filter for relevant subset
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset 'eo13960_scored' loaded. Rows: {len(subset)}")

# --- 2. Data Cleaning & Mapping ---

# Map Commercial Status based on debug findings
# 'None of the above.' -> Custom/Gov (0)
# Any other specific commercial description -> Commercial (1)
# NaN -> Exclude

def map_commercial_status(val):
    s = str(val).strip()
    if pd.isna(val) or s.lower() == 'nan':
        return np.nan
    if s == 'None of the above.':
        return 0
    return 1

subset['is_commercial'] = subset['10_commercial_ai'].apply(map_commercial_status)

# Drop rows where commercial status is ambiguous (NaN)
analysis_df = subset.dropna(subset=['is_commercial']).copy()

# Map Transparency Controls to Binary (1/0)
def map_binary(val):
    s = str(val).lower().strip()
    if s in ['yes', 'true', '1', '1.0']:
        return 1
    return 0

analysis_df['has_data_docs'] = analysis_df['34_data_docs'].apply(map_binary)
analysis_df['has_code_access'] = analysis_df['38_code_access'].apply(map_binary)

# Create label column
analysis_df['source_label'] = analysis_df['is_commercial'].map({1: 'Commercial (COTS)', 0: 'Custom / Gov'})

print("\n--- Analysis Groups ---")
print(analysis_df['source_label'].value_counts())

# --- 3. Statistical Analysis ---

results = {}

for control_col, label in [('has_data_docs', 'Data Documentation'), ('has_code_access', 'Code Access')]:
    print(f"\n>>> Analyzing: {label} <<<")
    
    # Contingency Table
    contingency = pd.crosstab(analysis_df['source_label'], analysis_df[control_col])
    
    # Ensure 0 (Absent) and 1 (Present) columns exist
    for c in [0, 1]:
        if c not in contingency.columns:
            contingency[c] = 0
    contingency = contingency[[0, 1]]
    
    print("Contingency Table (0=Absent, 1=Present):")
    print(contingency)
    
    # Compliance Rates
    rates = analysis_df.groupby('source_label')[control_col].mean()
    print(f"Compliance Rates:\n{rates}")
    
    # Chi-Square Test
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"Chi-Square: {chi2:.4f}, p-value: {p:.4e}")
    
    # Odds Ratio Calculation
    # OR = (Comm_Present / Comm_Absent) / (Cust_Present / Cust_Absent)
    # To avoid division by zero, we add 0.5 to cells if any cell is 0 (Haldane-Anscombe correction)
    # but for simple reporting we can check directly.
    
    try:
        comm_p = contingency.loc['Commercial (COTS)', 1]
        comm_a = contingency.loc['Commercial (COTS)', 0]
        cust_p = contingency.loc['Custom / Gov', 1]
        cust_a = contingency.loc['Custom / Gov', 0]
        
        # Use correction if needed for infinity
        if comm_a == 0 or cust_p == 0 or cust_a == 0 or comm_p == 0:
             odds_cots = (comm_p + 0.5) / (comm_a + 0.5)
             odds_cust = (cust_p + 0.5) / (cust_a + 0.5)
        else:
             odds_cots = comm_p / comm_a
             odds_cust = cust_p / cust_a
            
        or_val = odds_cots / odds_cust
        print(f"Odds Ratio (Commercial / Custom): {or_val:.4f}")
        
        if p < 0.05:
            if or_val < 1:
                print(f"SIGNIFICANT: Commercial systems are {(1/or_val):.2f}x LESS likely to have {label}.")
            else:
                print(f"SIGNIFICANT: Commercial systems are {or_val:.2f}x MORE likely to have {label}.")
        else:
            print("Result: Not statistically significant.")
            
    except Exception as e:
        print(f"Error calculating odds ratio: {e}")
        
    results[label] = rates

# --- 4. Visualization ---

fig, ax = plt.subplots(figsize=(10, 6))
groups = ['Commercial (COTS)', 'Custom / Gov']
x = np.arange(len(groups))
width = 0.35

# Get values in correct order
docs_vals = [results['Data Documentation'].get(g, 0) * 100 for g in groups]
code_vals = [results['Code Access'].get(g, 0) * 100 for g in groups]

rects1 = ax.bar(x - width/2, docs_vals, width, label='Data Documentation', color='#4e79a7')
rects2 = ax.bar(x + width/2, code_vals, width, label='Code Access', color='#f28e2b')

ax.set_ylabel('Compliance Rate (%)')
ax.set_title('Transparency Gap: Commercial vs Custom AI (Federal Inventory)')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, max(max(docs_vals), max(code_vals)) * 1.3 if max(code_vals) > 0 else 10)
plt.tight_layout()
plt.show()
