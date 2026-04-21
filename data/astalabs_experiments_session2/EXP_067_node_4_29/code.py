import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import sys

# --- 1. Load Dataset ---
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Subset Shape: {eo_df.shape}")

# --- 2. Define Groups using '37_custom_code' ---
# Hypothesis: 'No' Custom Code -> Commercial (COTS), 'Yes' Custom Code -> Custom (Gov)
group_col = '37_custom_code'
valid_groups = ['Yes', 'No']

# Filter data to only include valid 'Yes'/'No' responses
analysis_df = eo_df[eo_df[group_col].isin(valid_groups)].copy()

# Map to readable labels
analysis_df['group'] = analysis_df[group_col].map({'No': 'Commercial (COTS)', 'Yes': 'Custom (Gov)'})
print(f"Analysis Subset Shape: {analysis_df.shape}")
print(analysis_df['group'].value_counts())

# --- 3. Statistical Analysis ---
targets = {
    'Transparency: Code Access': '38_code_access',
    'Transparency: Data Docs': '34_data_docs',
    'Security: ATO': '40_has_ato'
}

def normalize_response(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    # Treat definitive affirmative answers as 1
    if val_str in ['yes', 'y', 'true', '1']:
        return 1
    return 0

stats_results = []
plot_data = {}

print("\n--- Chi-Square Test Results ---")

for label, col in targets.items():
    if col not in analysis_df.columns:
        print(f"Warning: Column {col} not found. Skipping.")
        continue
        
    # Normalize
    analysis_df[f'clean_{col}'] = analysis_df[col].apply(normalize_response)
    
    # Create Contingency Table
    # Rows: Groups, Cols: Compliance (0, 1)
    contingency = pd.crosstab(analysis_df['group'], analysis_df[f'clean_{col}'])
    
    # Ensure 0 and 1 columns exist
    for c in [0, 1]:
        if c not in contingency.columns:
            contingency[c] = 0
            
    # Calculate percentages
    totals = contingency.sum(axis=1)
    # contingency[1] is the count of 'Yes'
    compliance_rates = (contingency[1] / totals * 100)
    
    # Chi-square test
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    
    stats_results.append({
        'Control': label,
        'Comm_Rate': compliance_rates.get('Commercial (COTS)', 0),
        'Custom_Rate': compliance_rates.get('Custom (Gov)', 0),
        'p_value': p
    })
    
    plot_data[label] = compliance_rates
    
    print(f"\nControl: {label}")
    print(contingency)
    print(f"  Commercial (COTS) Compliance: {compliance_rates.get('Commercial (COTS)', 0):.1f}%")
    print(f"  Custom (Gov) Compliance:      {compliance_rates.get('Custom (Gov)', 0):.1f}%")
    print(f"  p-value: {p:.4e}")
    if p < 0.05:
        print("  -> Statistically Significant")
    else:
        print("  -> Not Significant")

# --- 4. Visualization ---
results_df = pd.DataFrame(stats_results)
if not results_df.empty:
    labels = [r['Control'] for r in stats_results]
    comm_vals = [r['Comm_Rate'] for r in stats_results]
    cust_vals = [r['Custom_Rate'] for r in stats_results]
    p_vals = [r['p_value'] for r in stats_results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, comm_vals, width, label='Commercial (COTS)', color='#ff7f0e')
    rects2 = ax.bar(x + width/2, cust_vals, width, label='Custom (Gov)', color='#1f77b4')

    ax.set_ylabel('Compliance Rate (%)')
    ax.set_title('Transparency vs Security: Commercial vs Custom AI (EO 13960)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add p-values
    for i, p in enumerate(p_vals):
        height = max(comm_vals[i], cust_vals[i]) + 3
        p_text = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(i, height, p_text, ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()
else:
    print("No results to plot.")