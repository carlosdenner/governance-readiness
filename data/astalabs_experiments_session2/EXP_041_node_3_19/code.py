import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for EO13960 data subset
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- 1. Map Development Stage ---
def map_stage(stage):
    if pd.isna(stage):
        return np.nan
    s = str(stage).lower().strip()
    
    # Legacy / O&M
    if any(x in s for x in ['operation', 'production', 'mission']):
        return 'Legacy (O&M)'
    
    # New / Development
    if any(x in s for x in ['acquisition', 'development', 'initiated', 'implementation', 'planned']):
        return 'New (Dev/Plan)'
        
    return np.nan

eo_df['Stage_Group'] = eo_df['16_dev_stage'].apply(map_stage)

# --- 2. Map Controls to Binary ---

def map_monitoring(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    # Negatives
    if 'no monitoring' in s or 'not safety' in s:
        return 0
    
    # Positives (Intermittent, Automated, Established, Under development)
    if any(x in s for x in ['intermittent', 'automated', 'established', 'under development']):
        return 1
        
    return np.nan

def map_testing(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    # Positives: Real-world / Operational environment
    if 'performance evaluation' in s or 'impact evaluation' in s or s == 'yes':
        return 1
    
    # Negatives: No testing, Benchmark only (explicitly not operational), Waived
    if 'no testing' in s or 'benchmark' in s or 'waived' in s or 'not safety' in s:
        return 0
        
    return np.nan

eo_df['Monitor_Bin'] = eo_df['56_monitor_postdeploy'].apply(map_monitoring)
eo_df['Test_Bin'] = eo_df['53_real_world_testing'].apply(map_testing)

# --- 3. Analysis Loop ---
controls = {
    'Monitor_Bin': 'Monitoring Post-Deployment',
    'Test_Bin': 'Real-World Testing'
}

results = []

for col_var, col_label in controls.items():
    # Filter for rows that have both a Stage and a valid Answer (0 or 1)
    subset = eo_df.dropna(subset=['Stage_Group', col_var]).copy()
    
    if len(subset) == 0:
        print(f"No valid data for {col_label}")
        continue
        
    contingency = pd.crosstab(subset['Stage_Group'], subset[col_var])
    
    # Check if we have enough data dimensions
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        print(f"Insufficient variance for {col_label}. Shape: {contingency.shape}")
        print(contingency)
        continue
        
    chi2, p, dof, ex = chi2_contingency(contingency)
    
    # Calculate Compliance Rates
    rates = subset.groupby('Stage_Group')[col_var].mean()
    counts = subset.groupby('Stage_Group')[col_var].count()
    
    legacy_rate = rates.get('Legacy (O&M)', 0)
    new_rate = rates.get('New (Dev/Plan)', 0)
    
    results.append({
        'Control': col_label,
        'Legacy_Rate': legacy_rate,
        'New_Rate': new_rate,
        'p_value': p,
        'Legacy_N': counts.get('Legacy (O&M)', 0),
        'New_N': counts.get('New (Dev/Plan)', 0)
    })
    
    print(f"\n--- Analysis: {col_label} ---")
    print(contingency)
    print(f"Legacy Compliance: {legacy_rate:.2%} (n={counts.get('Legacy (O&M)', 0)})")
    print(f"New    Compliance: {new_rate:.2%} (n={counts.get('New (Dev/Plan)', 0)})")
    print(f"Chi-Square p-value: {p:.5f}")

# --- 4. Plotting ---
if results:
    res_df = pd.DataFrame(results)
    
    x = np.arange(len(res_df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, res_df['Legacy_Rate'], width, label='Legacy (O&M)', color='indianred')
    rects2 = ax.bar(x + width/2, res_df['New_Rate'], width, label='New (Dev/Plan)', color='steelblue')
    
    ax.set_ylabel('Compliance Rate (1=Yes, 0=No/Partial)')
    ax.set_title('Governance Compliance by Development Stage (EO 13960)')
    ax.set_xticks(x)
    ax.set_xticklabels(res_df['Control'])
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    # Annotate p-values
    for i, row in res_df.iterrows():
        max_h = max(row['Legacy_Rate'], row['New_Rate'])
        sig_text = "*" if row['p_value'] < 0.05 else "ns"
        ax.text(i, max_h + 0.05, f"p={row['p_value']:.3f}\n({sig_text})", ha='center')
        
    plt.tight_layout()
    plt.show()
    
    # Interpret results
    print("\n--- Conclusion ---")
    for i, row in res_df.iterrows():
        if row['p_value'] < 0.05:
            direction = "Lower" if row['Legacy_Rate'] < row['New_Rate'] else "Higher"
            print(f"{row['Control']}: Significant difference. Legacy systems show {direction} compliance.")
        else:
            print(f"{row['Control']}: No significant difference detected.")
