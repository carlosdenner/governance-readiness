import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import numpy as np

# Define file path
file_name = 'astalabs_discovery_all_data.csv'
file_path = f'../{file_name}'
if not os.path.exists(file_path):
    file_path = file_name

# Load dataset
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: Could not find {file_name}")
    exit(1)

# Filter for EO13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- 1. Define Independent Variable: Procurement Source ---
# Column: '22_dev_method'

def classify_source(val):
    s = str(val).lower().strip()
    is_contracting = 'contracting' in s
    is_in_house = 'in-house' in s
    
    if is_contracting and not is_in_house:
        return 'Commercial'
    elif is_in_house and not is_contracting:
        return 'Internal'
    else:
        return None

eo_df['procurement_source'] = eo_df['22_dev_method'].apply(classify_source)

# Drop rows where source is undefined
analysis_df = eo_df.dropna(subset=['procurement_source']).copy()

# --- 2. Define Dependent Variable: Documentation Availability ---
# Column: '34_data_docs'

def check_docs(val):
    if pd.isna(val):
        return False
    s = str(val).lower().strip()
    # explicit negatives
    if 'missing' in s or 'not available' in s or 'not reported' in s or s == 'no' or s == '':
        return False
    # explicit positives
    if 'complete' in s or 'available' in s or 'partial' in s or 'yes' in s or 'public' in s:
        return True
    return False

analysis_df['has_docs'] = analysis_df['34_data_docs'].apply(check_docs)

# --- 3. Analysis ---

# Contingency Table
raw_contingency = pd.crosstab(analysis_df['procurement_source'], analysis_df['has_docs'])

# Robustly ensure 2x2 shape using reindex
# Index: Internal first (reference), then Commercial
# Columns: False (No Docs), True (Has Docs)
contingency = raw_contingency.reindex(index=['Internal', 'Commercial'], columns=[False, True], fill_value=0)
contingency.columns = ['No Docs', 'Has Docs']

print("--- Contingency Table (Source vs Documentation) ---")
print(contingency)

# Calculate Rates
internal_total = contingency.loc['Internal'].sum()
commercial_total = contingency.loc['Commercial'].sum()

if internal_total > 0:
    int_rate = (contingency.loc['Internal', 'Has Docs'] / internal_total) * 100
else:
    int_rate = 0

if commercial_total > 0:
    comm_rate = (contingency.loc['Commercial', 'Has Docs'] / commercial_total) * 100
else:
    comm_rate = 0

print(f"\nInternal Systems with Docs:   {int_rate:.2f}% (N={internal_total})")
print(f"Commercial Systems with Docs: {comm_rate:.2f}% (N={commercial_total})")

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpretation
alpha = 0.05
print(f"\nInterpretation (alpha={alpha}):")
if p < alpha:
    print("Result: Statistically SIGNIFICANT difference found.")
    if comm_rate < int_rate:
        print("Hypothesis SUPPORTED: Commercial systems have significantly LOWER documentation rates.")
    else:
        print("Hypothesis REFUTED: Commercial systems have significantly HIGHER documentation rates.")
else:
    print("Result: NO statistically significant difference found.")

# Odds Ratio Calculation
# OR = (Odds of Docs given Commercial) / (Odds of Docs given Internal)
# Odds = P / (1-P)
odds_comm = comm_rate / (100 - comm_rate) if comm_rate != 100 else np.inf
odds_int = int_rate / (100 - int_rate) if int_rate != 100 else np.inf

if odds_int == 0:
    or_val = np.inf
else:
    or_val = odds_comm / odds_int
    
print(f"Odds Ratio (Commercial / Internal): {or_val:.4f}")

# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
# Normalize to get percentages
props = contingency.div(contingency.sum(axis=1), axis=0)

ax = props.plot(kind='bar', stacked=True, color=['#d9534f', '#5bc0de'], ax=plt.gca())

plt.title('Data Documentation Availability: Commercial vs. Internal AI', fontsize=14)
plt.xlabel('Procurement Source', fontsize=12)
plt.ylabel('Proportion of Systems', fontsize=12)
plt.ylim(0, 1.15) # Extra space for labels
plt.legend(title='Documentation Status', loc='upper right')
plt.xticks(rotation=0)

# Annotate bars
for c in ax.containers:
    # Only label if segment is big enough
    labels = [f'{v.get_height()*100:.1f}%' if v.get_height() > 0.05 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', fontweight='bold')

plt.tight_layout()
plt.show()