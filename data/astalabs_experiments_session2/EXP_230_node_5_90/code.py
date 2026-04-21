import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
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

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# refined mapping logic based on previous output
def map_impact_level(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    
    # High Impact: Rights, Safety, Both, High, Significant
    if any(x in val_lower for x in ['rights', 'safety', 'both', 'high', 'significant', 'critical']):
        return 'High Impact'
    # Low Impact: Neither, Low, Moderate, Minimal
    # Note: 'Neither' in EO 13960 context means neither safety nor rights impacting, effectively 'Low' for this binary.
    elif any(x in val_lower for x in ['neither', 'low', 'moderate', 'minimal', 'minor']):
        return 'Low Impact'
    
    return 'Uncategorized'

def map_evaluation_evidence(val):
    if pd.isna(val):
        return 0
    val_lower = str(val).lower()
    
    # Affirmative keywords
    affirmative = ['yes', 'conducted', 'completed', 'performed', 'ongoing', '3rd party', 'third party', 'independent', 'true']
    
    if any(x in val_lower for x in affirmative):
        # Exclude negations if necessary. 
        # Specific check for 'not conducted' or 'does not apply' if 'yes' is absent, 
        # but sometimes 'yes' appears in a sentence saying 'yes we did it'.
        # However, 'yes - by the caio' is a positive.
        # 'planned' is not done yet, so usually 0, but let's be strict: has it been done?
        if 'planned' in val_lower and 'completed' not in val_lower and 'ongoing' not in val_lower:
             return 0
        return 1
    return 0

# Apply Mappings
eo_df['impact_group'] = eo_df['17_impact_type'].apply(map_impact_level)
eo_df['has_eval'] = eo_df['55_independent_eval'].apply(map_evaluation_evidence)

# Check distribution
print("Impact Group Distribution:")
print(eo_df['impact_group'].value_counts())

# Filter for analysis groups
analysis_df = eo_df[eo_df['impact_group'].isin(['High Impact', 'Low Impact'])].copy()

# Calculate Statistics
group_stats = analysis_df.groupby('impact_group')['has_eval'].agg(['count', 'sum', 'mean'])
group_stats['pct'] = group_stats['mean'] * 100

print("\n--- Comparative Statistics ---")
print(group_stats)

# Ensure we have both groups
if 'High Impact' not in group_stats.index or 'Low Impact' not in group_stats.index:
    print("\nError: Missing one of the comparison groups (High Impact or Low Impact). Aborting Z-test.")
else:
    # Statistical Test (Z-test)
    high_grp = group_stats.loc['High Impact']
    low_grp = group_stats.loc['Low Impact']

    count_arr = np.array([high_grp['sum'], low_grp['sum']])
    nobs_arr = np.array([high_grp['count'], low_grp['count']])

    z_score, p_value = proportions_ztest(count_arr, nobs_arr, alternative='two-sided')

    print(f"\nZ-Score: {z_score:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    print(f"Result: {significance}")

    # Visualization
    plt.figure(figsize=(10, 6))
    bar_colors = ['#d62728', '#1f77b4'] # Red for High, Blue for Low (alphabetical order H, L)
    
    # Create bar plot
    # Note: groupby sorts alphabetically by index: High Impact, Low Impact. 
    # High Impact (H) comes before Low Impact (L)? No, H comes before L.
    bars = plt.bar(group_stats.index, group_stats['pct'], color=bar_colors, alpha=0.8)

    # Error bars
    se = np.sqrt(group_stats['mean'] * (1 - group_stats['mean']) / group_stats['count']) * 100
    plt.errorbar(group_stats.index, group_stats['pct'], yerr=se, fmt='none', ecolor='black', capsize=5)

    plt.title('Independent Evaluation Rates: High vs Low Impact AI')
    plt.ylabel('Percentage with Independent Evaluation (%)')
    plt.xlabel('Impact Classification')
    plt.ylim(0, max(group_stats['pct']) * 1.5 if len(group_stats) > 0 else 10)

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontweight='bold')

    stats_text = (f"Z-test: z={z_score:.2f}, p={p_value:.3f}\n"
                  f"n(High)={int(high_grp['count'])}, n(Low)={int(low_grp['count'])}")
    plt.text(0.5, 0.9, stats_text, transform=plt.gca().transAxes, 
             ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.show()