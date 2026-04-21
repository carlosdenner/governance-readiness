import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for eo13960_scored
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Normalize agency names for grouping
df_eo['agency_norm'] = df_eo['3_agency'].fillna('').astype(str).str.lower()

# Define groups
security_keywords = ['defense', 'homeland', 'justice', 'state']
social_keywords = ['health', 'education', 'labor', 'housing', 'veterans']

def categorize_agency(name):
    if any(k in name for k in security_keywords):
        return 'Security'
    elif any(k in name for k in social_keywords):
        return 'Social'
    return 'Other'

df_eo['Agency_Group'] = df_eo['agency_norm'].apply(categorize_agency)

# Filter for only Security and Social groups
df_analysis = df_eo[df_eo['Agency_Group'].isin(['Security', 'Social'])].copy()

# Identify correct columns dynamically to avoid KeyErrors
ato_cols = [c for c in df_eo.columns if '40_has_ato' in c]
rights_cols = [c for c in df_eo.columns if 'disparity_mitigation' in c]

if not ato_cols or not rights_cols:
    print(f"Could not find required columns. Available columns related to ATO: {ato_cols}, Rights: {rights_cols}")
    # Fallback search if exact partial match fails
    ato_cols = [c for c in df_eo.columns if 'ato' in c.lower()]
    rights_cols = [c for c in df_eo.columns if 'disparity' in c.lower()]

ato_col = ato_cols[0]
rights_col = rights_cols[0]

print(f"Using Security Column: {ato_col}")
print(f"Using Rights Column: {rights_col}")

# Clean control columns
def clean_binary(val):
    s = str(val).lower()
    if s in ['yes', 'true', '1', '1.0']:
        return 1
    return 0

df_analysis['Security_Control'] = df_analysis[ato_col].apply(clean_binary)
df_analysis['Rights_Control'] = df_analysis[rights_col].apply(clean_binary)

# Generate summary stats
group_stats = df_analysis.groupby('Agency_Group').agg(
    n=('source_row_num', 'count'),
    security_count=('Security_Control', 'sum'),
    rights_count=('Rights_Control', 'sum'),
    security_rate=('Security_Control', 'mean'),
    rights_rate=('Rights_Control', 'mean')
)

print("Summary Statistics:")
print(group_stats)
print("\n")

# Statistical Tests (Two-Proportion Z-test)
# We compare 'Social' vs 'Security' for both control types

results = {}

for control_type, count_col, nobs_col in [('Security_Control', 'security_count', 'n'), ('Rights_Control', 'rights_count', 'n')]:
    counts = np.array([group_stats.loc['Social', count_col], group_stats.loc['Security', count_col]])
    nobs = np.array([group_stats.loc['Social', nobs_col], group_stats.loc['Security', nobs_col]])
    
    # Handle cases with 0 variance or small sample size if necessary, but ztest usually handles it unless n=0
    stat, pval = proportions_ztest(counts, nobs)
    results[control_type] = (stat, pval)
    print(f"Z-test for {control_type} (Social vs Security): z={stat:.4f}, p={pval:.4f}")

# Calculate 'Prioritization Gap' (Security - Rights) within each group
group_stats['Prioritization_Gap'] = group_stats['security_rate'] - group_stats['rights_rate']
print("\nPrioritization Gap (Security Rate - Rights Rate):")
print(group_stats['Prioritization_Gap'])

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(group_stats))

bar1 = ax.bar(index, group_stats['security_rate'], bar_width, label='Security (ATO)', alpha=0.8)
bar2 = ax.bar(index + bar_width, group_stats['rights_rate'], bar_width, label='Rights (Bias Mitigation)', alpha=0.8)

ax.set_xlabel('Agency Type')
ax.set_ylabel('Prevalence of Control')
ax.set_title('Governance Priorities: Security vs Social Agencies')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(group_stats.index)
ax.legend()

plt.tight_layout()
plt.show()
