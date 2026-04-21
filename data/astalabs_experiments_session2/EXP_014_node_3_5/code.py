import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess

# Install statsmodels if not present
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    from statsmodels.stats.proportion import proportions_ztest
except ImportError:
    install("statsmodels")
    from statsmodels.stats.proportion import proportions_ztest

# 1. Load Dataset
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from {file_path}...")
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    sys.exit(1)

# 2. Filter for EO 13960 Scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset loaded. Shape: {df_eo.shape}")

# 3. Define and Apply Stage Grouping
stage_col = '16_dev_stage'
def categorize_stage(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).lower().strip()
    if any(x in val_str for x in ['operation', 'use', 'deployed']):
        return 'Operational'
    if any(x in val_str for x in ['develop', 'test', 'pilot']):
        return 'Development/Testing'
    return 'Other'

df_eo['stage_group'] = df_eo[stage_col].apply(categorize_stage)

# Create groups
group_ops = df_eo[df_eo['stage_group'] == 'Operational'].copy()
group_dev = df_eo[df_eo['stage_group'] == 'Development/Testing'].copy()

print(f"\nOperational Group Size: {len(group_ops)}")
print(f"Development/Testing Group Size: {len(group_dev)}")

if len(group_ops) == 0 or len(group_dev) == 0:
    print("Error: Insufficient data in one or both groups to perform analysis.")
    sys.exit(0)

# 4. Text Analysis Mapping Functions
def map_monitoring(val):
    if pd.isna(val):
        return 0
    v = str(val).lower()
    # Positive indicators based on debug output
    if any(x in v for x in ['intermittent', 'automated', 'established process', 'manually updated']):
        return 1
    # Negative indicators: 'no monitoring', 'not safety', 'under development'
    return 0

def map_iqa(val):
    if pd.isna(val):
        return 0
    v = str(val).lower()
    # Negative indicators first
    if any(x in v for x in ['not applicable', 'n/a', 'non-public', 'proof of concept', 'no iqa', 'not in production']):
        return 0
    # Positive indicators: check for presence of meaningful text
    # If it's not explicitly N/A and has significant length, assume some compliance is cited
    if len(v) > 10:
        return 1
    return 0

def map_risks(val):
    if pd.isna(val):
        return 0
    v = str(val).lower()
    # Negative indicators
    if any(x in v for x in ['forthcoming', 'not applicable', 'no key risks', 'n/a', 'not safety']):
        return 0
    # Positive indicators
    if len(v) > 10:
        return 1
    return 0

# Apply mappings
metrics = {
    '56_monitor_postdeploy': {'label': 'Post-Deploy Monitoring', 'func': map_monitoring},
    '28_iqa_compliance': {'label': 'IQA Compliance', 'func': map_iqa},
    '54_key_risks': {'label': 'Risk Assessment', 'func': map_risks}
}

results = []
print("\n--- Governance Control Analysis (Mapped) ---")

for col, meta in metrics.items():
    label = meta['label']
    func = meta['func']
    
    if col not in df_eo.columns:
        print(f"Skipping {label}: Column {col} not found.")
        continue
        
    # Calculate counts
    ops_success = group_ops[col].apply(func).sum()
    ops_total = len(group_ops)
    
    dev_success = group_dev[col].apply(func).sum()
    dev_total = len(group_dev)
    
    ops_prop = ops_success / ops_total if ops_total > 0 else 0
    dev_prop = dev_success / dev_total if dev_total > 0 else 0
    
    print(f"\nAnalyzing '{label}'")
    print(f"  Operational: {ops_success}/{ops_total} ({ops_prop:.2%})")
    print(f"  Dev/Testing: {dev_success}/{dev_total} ({dev_prop:.2%})")
    
    # Z-test
    stat, pval = 0.0, 1.0
    significant = False
    if (ops_success > 0 or dev_success > 0) and (ops_success < ops_total or dev_success < dev_total):
        stat, pval = proportions_ztest([ops_success, dev_success], [ops_total, dev_total])
        significant = pval < 0.05
        print(f"  Z-test: z={stat:.4f}, p={pval:.4e}")
    else:
        print("  Z-test: Skipped (no variance)")

    # Interpretation
    if significant:
        if ops_prop < dev_prop:
            interp = "DECAY DETECTED (Ops < Dev)"
        else:
            interp = "MATURATION DETECTED (Ops > Dev)"
    else:
        interp = "No Significant Difference"
    print(f"  Result: {interp}")

    results.append({
        'Metric': label,
        'Ops_Prop': ops_prop,
        'Dev_Prop': dev_prop,
        'P_Value': pval,
        'Significant': significant
    })

# 5. Visualization
if results:
    res_df = pd.DataFrame(results)
    
    x = np.arange(len(res_df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, res_df['Ops_Prop'], width, label='Operational', color='#4c72b0')
    rects2 = ax.bar(x + width/2, res_df['Dev_Prop'], width, label='Dev/Testing', color='#dd8452')
    
    ax.set_ylabel('Active Governance (Proportion)')
    ax.set_title('Governance Lifecycle Analysis: Operational vs Development')
    ax.set_xticks(x)
    ax.set_xticklabels(res_df['Metric'])
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    
    # Add p-values
    for i, row in res_df.iterrows():
        h = max(row['Ops_Prop'], row['Dev_Prop'])
        txt = f"p={row['P_Value']:.3f}"
        if row['Significant']: txt += " *"
        ax.text(i, h + 0.05, txt, ha='center', fontweight='bold' if row['Significant'] else 'normal')

    plt.tight_layout()
    plt.show()