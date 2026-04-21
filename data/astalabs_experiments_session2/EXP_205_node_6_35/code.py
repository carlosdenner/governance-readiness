import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

# [debug]
print("Starting execution...")

# 1. Load Dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 scored data
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {subset.shape}")

# 2. Define Procurement Groups
# Using '10_commercial_ai' as the primary differentiator for Commercial (COTS) vs Custom/Other.
# 'None of the above.' implies the system does not fall into the specific commercial use-case categories.
# We treat 'None of the above.' as the Control group (Likely Custom/GOTS/Internal).

def define_procurement(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == 'None of the above.':
        return 'Custom/GOTS'
    else:
        # Any specific commercial use case description
        return 'Commercial (COTS)'

subset['procurement_type'] = subset['10_commercial_ai'].apply(define_procurement)

# 3. Clean Transparency Columns

def clean_code_access(val):
    if pd.isna(val):
        return None
    s = str(val).lower()
    # Check for negative assertions
    if s.startswith('no') or 'no access' in s:
        return 'No'
    # Check for positive assertions
    if s.startswith('yes') or 'available' in s or 'public' in s:
        return 'Yes'
    return None

def clean_data_docs(val):
    if pd.isna(val):
        return None
    s = str(val).lower()
    # Negative
    if 'missing' in s or 'not available' in s:
        return 'No'
    # Positive
    if 'complete' in s or 'partial' in s or 'widely' in s or 'exists' in s:
        return 'Yes'
    return None

subset['code_access_bin'] = subset['38_code_access'].apply(clean_code_access)
subset['data_docs_bin'] = subset['34_data_docs'].apply(clean_data_docs)

# Filter to analyzable rows (Must have procurement type)
clean_df = subset.dropna(subset=['procurement_type'])
print(f"Analyzable rows: {len(clean_df)}")
print("\nGroup Sizes:")
print(clean_df['procurement_type'].value_counts())

# 4. Statistical Analysis & Visualization Setup
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
results_found = False

# --- Analysis A: Code Access ---
print("\n--- Analysis: Commercial vs Code Access ---")
df_code = clean_df.dropna(subset=['code_access_bin'])
if len(df_code) > 0:
    results_found = True
    ct_code = pd.crosstab(df_code['procurement_type'], df_code['code_access_bin'])
    print(ct_code)
    
    # Chi-Square
    chi2, p, dof, ex = stats.chi2_contingency(ct_code)
    print(f"Chi-Square: {chi2:.4f}, p-value: {p:.4e}")
    
    # Odds Ratio (Odds of NO Access)
    if 'Commercial (COTS)' in ct_code.index and 'Custom/GOTS' in ct_code.index and 'No' in ct_code.columns:
        c_no = ct_code.loc['Commercial (COTS)', 'No']
        c_yes = ct_code.loc['Commercial (COTS)', 'Yes']
        g_no = ct_code.loc['Custom/GOTS', 'No']
        g_yes = ct_code.loc['Custom/GOTS', 'Yes']
        
        # Laplace smoothing if needed
        if any(x==0 for x in [c_no, c_yes, g_no, g_yes]):
             c_no+=0.5; c_yes+=0.5; g_no+=0.5; g_yes+=0.5
             
        or_val = (c_no / c_yes) / (g_no / g_yes)
        print(f"Odds Ratio (Commercial likelihood of NO access): {or_val:.4f}")
    
    # Plot
    ct_norm = pd.crosstab(df_code['procurement_type'], df_code['code_access_bin'], normalize='index')
    colors = ['#d62728', '#2ca02c'] if 'No' == ct_norm.columns[0] else ['#2ca02c', '#d62728']
    ct_norm.plot(kind='bar', stacked=True, ax=axes[0], color=colors)
    axes[0].set_title(f"Code Access Transparency\n(n={len(df_code)})\np={p:.1e}")
    axes[0].set_ylabel("Proportion")
    axes[0].tick_params(axis='x', rotation=0)

# --- Analysis B: Data Documentation ---
print("\n--- Analysis: Commercial vs Data Documentation ---")
df_docs = clean_df.dropna(subset=['data_docs_bin'])
if len(df_docs) > 0:
    results_found = True
    ct_docs = pd.crosstab(df_docs['procurement_type'], df_docs['data_docs_bin'])
    print(ct_docs)
    
    chi2, p, dof, ex = stats.chi2_contingency(ct_docs)
    print(f"Chi-Square: {chi2:.4f}, p-value: {p:.4e}")
    
    # Odds Ratio
    if 'Commercial (COTS)' in ct_docs.index and 'Custom/GOTS' in ct_docs.index and 'No' in ct_docs.columns:
        c_no = ct_docs.loc['Commercial (COTS)', 'No']
        c_yes = ct_docs.loc['Commercial (COTS)', 'Yes']
        g_no = ct_docs.loc['Custom/GOTS', 'No']
        g_yes = ct_docs.loc['Custom/GOTS', 'Yes']
        
        if any(x==0 for x in [c_no, c_yes, g_no, g_yes]):
             c_no+=0.5; c_yes+=0.5; g_no+=0.5; g_yes+=0.5
             
        or_val = (c_no / c_yes) / (g_no / g_yes)
        print(f"Odds Ratio (Commercial likelihood of NO docs): {or_val:.4f}")

    # Plot
    ct_norm = pd.crosstab(df_docs['procurement_type'], df_docs['data_docs_bin'], normalize='index')
    colors = ['#d62728', '#2ca02c'] if 'No' == ct_norm.columns[0] else ['#2ca02c', '#d62728']
    ct_norm.plot(kind='bar', stacked=True, ax=axes[1], color=colors)
    axes[1].set_title(f"Data Documentation Transparency\n(n={len(df_docs)})\np={p:.1e}")
    axes[1].set_ylabel("Proportion")
    axes[1].tick_params(axis='x', rotation=0)

plt.suptitle("Commercial (COTS) vs Custom (GOTS) Transparency Gap", fontsize=16)
plt.tight_layout()
if results_found:
    plt.show()
else:
    print("No valid data to plot.")
