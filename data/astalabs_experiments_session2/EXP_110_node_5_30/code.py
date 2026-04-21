import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# --- 1. Load Dataset ---
candidates = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
file_path = None
for c in candidates:
    if os.path.exists(c):
        file_path = c
        break

if file_path is None:
    file_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from: {file_path}")
df_all = pd.read_csv(file_path, low_memory=False)

# --- 2. Data Preparation ---
df_aiid = df_all[df_all['source_table'] == 'aiid_incidents'].copy()

intent_col = 'Intentional Harm'
sector_col = 'Sector of Deployment'

# --- 3. Improved Mapping Logic ---
# Based on previous output, values start with 'Yes.' or 'No.'
def is_intentional(val):
    s = str(val).lower().strip()
    # Check for explicit affirmative start or keyword
    if s.startswith('yes') or 'intentionally designed' in s:
        return 1
    return 0

df_aiid['is_intentional'] = df_aiid[intent_col].apply(is_intentional)

# Clean Sector and remove 'Unknown' for better analysis
df_aiid['sector_clean'] = df_aiid[sector_col].fillna('Unknown').astype(str)
df_analyzable = df_aiid[df_aiid['sector_clean'].str.lower() != 'unknown'].copy()

# --- 4. Filter Top 5 Known Sectors ---
top_sectors = df_analyzable['sector_clean'].value_counts().nlargest(5).index.tolist()
df_top = df_analyzable[df_analyzable['sector_clean'].isin(top_sectors)].copy()

print(f"\nTop 5 Known Sectors: {top_sectors}")

# --- 5. Contingency Table ---
contingency = pd.crosstab(df_top['sector_clean'], df_top['is_intentional'])

# Ensure columns 0 and 1 exist
for c in [0, 1]:
    if c not in contingency.columns:
        contingency[c] = 0
contingency = contingency[[0, 1]]
contingency.columns = ['Unintentional', 'Intentional']

print("\nContingency Table (Observed):")
print(contingency)

# --- 6. Statistical Test ---
total_intentional = contingency['Intentional'].sum()

if total_intentional == 0:
    print("\n[!] Still no intentional incidents found in top sectors after logic update.")
else:
    # Perform Chi-Square Test
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    # Calculate Standardized Residuals
    observed_intent = contingency['Intentional']
    expected_intent = pd.Series(expected[:, 1], index=contingency.index)
    
    # Avoid division by zero in residuals (add small epsilon if expected is 0, though chi2 usually handles this)
    safe_expected = expected_intent.replace(0, 1e-9)
    std_residuals = (observed_intent - expected_intent) / np.sqrt(safe_expected)

    results_df = pd.DataFrame({
        'Total': contingency.sum(axis=1),
        'Intentional': observed_intent,
        'Rate': observed_intent / contingency.sum(axis=1),
        'Exp_Intent': expected_intent,
        'Residual': std_residuals
    }).sort_values('Residual', ascending=False)

    print("\n--- Chi-Square Results ---")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-Value:        {p_val:.4e}")
    
    print("\n--- Malicious Sector Profile (Ranked by Residual) ---")
    # Pretty print
    print_df = results_df.copy()
    print_df['Rate'] = print_df['Rate'].map('{:.1%}'.format)
    print_df['Residual'] = print_df['Residual'].map('{:.2f}'.format)
    print_df['Exp_Intent'] = print_df['Exp_Intent'].map('{:.1f}'.format)
    print(print_df[['Total', 'Intentional', 'Rate', 'Exp_Intent', 'Residual']].to_string())

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    # Normalize rows to 100%
    props = contingency.div(contingency.sum(axis=1), axis=0)
    # Sort by Intentional proportion
    props = props.sort_values('Intentional', ascending=True)
    
    ax = props.plot(kind='barh', stacked=True, color=['#A6CEE3', '#E31A1C'], figsize=(10, 6))
    plt.title('Proportion of Intentional vs Unintentional Harm by Top 5 Sectors')
    plt.xlabel('Proportion')
    plt.ylabel('Sector')
    plt.legend(['Unintentional', 'Intentional'], loc='lower right')
    plt.tight_layout()
    plt.show()