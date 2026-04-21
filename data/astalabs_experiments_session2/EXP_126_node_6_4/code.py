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
    except:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Loaded {len(eo_df)} records from EO 13960 source.")

col_reuse = '49_existing_reuse'
col_appeal = '65_appeal_process'

# --- MAPPING LOGIC ---

def map_system_type(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).lower().strip()
    
    # Explicit 'New' indicators
    if val_str.startswith('none') or val_str == 'no':
        return 'New (Custom)'
    
    # Explicit 'Legacy/Reuse' keywords found in unique values
    legacy_keywords = [
        're-use', 'reused', 'use of', 'built on', 'prior', 
        'shared', 'leveraged', 'used external'
    ]
    if any(k in val_str for k in legacy_keywords):
        return 'Legacy (Reused)'
    
    # If it's not explicitly None/No and contains description, treat as potential custom or ambiguous
    # But looking at unique values, most 'None' capture the custom ones.
    # Let's mark others as Unknown to be safe, or check if we missed any.
    return 'Unknown'

def map_appeal(val):
    if pd.isna(val):
        return 'No Appeal Process'
    val_str = str(val).lower().strip()
    if val_str.startswith('yes'):
        return 'Has Appeal Process'
    return 'No Appeal Process'

# Apply mappings
eo_df['system_type'] = eo_df[col_reuse].apply(map_system_type)
eo_df['appeal_status'] = eo_df[col_appeal].apply(map_appeal)

# Filter out Unknown system types
eo_df_clean = eo_df[eo_df['system_type'] != 'Unknown'].copy()

print("\n--- Group Counts ---")
print(eo_df_clean['system_type'].value_counts())

# Create Contingency Table
contingency_table = pd.crosstab(eo_df_clean['system_type'], eo_df_clean['appeal_status'])

print("\n--- Contingency Table ---")
print(contingency_table)

# Calculate Percentages
summary = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
print("\n--- Implementation Rates (% with Appeal Process) ---")
print(summary['Has Appeal Process'])

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant Difference (Reject Null Hypothesis)")
else:
    print("Result: No Statistically Significant Difference (Fail to Reject Null Hypothesis)")

# Visualization
ax = summary['Has Appeal Process'].plot(kind='bar', color=['orange', 'skyblue'], figsize=(8, 6))
plt.title('Appeal Process Implementation by System Origin')
plt.ylabel('Percentage with Appeal Process (%)')
plt.xlabel('System Origin')
plt.xticks(rotation=0)
plt.ylim(0, 20) # Focusing on the lower range if rates are low

for p_val, rect in zip(summary['Has Appeal Process'], ax.patches):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 0.5, f"{p_val:.1f}%", ha='center', va='bottom')

plt.tight_layout()
plt.show()
