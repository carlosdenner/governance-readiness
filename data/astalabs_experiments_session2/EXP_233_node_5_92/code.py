import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# [debug]
print("Starting experiment: Lifecycle Governance Decay")

# Load dataset
try:
    # Dataset files are one level above
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded EO 13960 data: {len(eo_df)} records")

# Check available columns to identify correct names
all_cols = eo_df.columns.tolist()
target_keywords = {
    'ato': 'has_ato',
    'impact': 'impact_assessment',
    'eval': 'independent_eval',
    'notice': 'ai_notice',
    'bias': 'disparity_mitigation'
}

found_cols = []
for key, keyword in target_keywords.items():
    matches = [c for c in all_cols if keyword in str(c).lower()]
    if matches:
        # Prefer the one that looks like the standard 'XX_name' format
        # Usually the shortest or the one appearing first is fine, but let's just take the first match
        found_cols.append(matches[0])
        print(f"Mapped '{key}' to column: {matches[0]}")
    else:
        print(f"Warning: Could not find column for '{key}'")

if not found_cols:
    print("Error: No governance columns found. Exiting.")
    exit(1)

# 2. Group 16_dev_stage
stage_col_matches = [c for c in all_cols if 'dev_stage' in str(c).lower()]
stage_col = stage_col_matches[0] if stage_col_matches else '16_dev_stage'
print(f"Using stage column: {stage_col}")

# Check unique values to define mapping
print("Unique development stages:", eo_df[stage_col].unique())

def classify_stage(val):
    s = str(val).lower()
    if any(x in s for x in ['plan', 'dev', 'design', 'acqui', 'research', 'pilot', 'test']):
        return 'Development'
    if any(x in s for x in ['oper', 'use', 'maint', 'deploy', 'implement', 'product']): 
        return 'Operation'
    return 'Other'

eo_df['lifecycle_group'] = eo_df[stage_col].apply(classify_stage)

# Filter out 'Other' or undefined
analysis_df = eo_df[eo_df['lifecycle_group'].isin(['Development', 'Operation'])].copy()
print(f"Records after stage filtering: {len(analysis_df)}")
print(analysis_df['lifecycle_group'].value_counts())

# 3. Create Governance Score
# Normalize binary values to 0/1
def normalize_binary(val):
    s = str(val).lower()
    if s in ['yes', 'true', '1', '1.0', 'y']:
        return 1
    return 0

for col in found_cols:
    analysis_df[col] = analysis_df[col].apply(normalize_binary)

analysis_df['governance_score'] = analysis_df[found_cols].sum(axis=1)

# 4. Statistical Analysis
dev_scores = analysis_df[analysis_df['lifecycle_group'] == 'Development']['governance_score']
ops_scores = analysis_df[analysis_df['lifecycle_group'] == 'Operation']['governance_score']

print(f"\n--- Results ---")
print(f"Development: Mean Score = {dev_scores.mean():.2f}, Median = {dev_scores.median()}, n = {len(dev_scores)}")
print(f"Operation:   Mean Score = {ops_scores.mean():.2f}, Median = {ops_scores.median()}, n = {len(ops_scores)}")

u_stat, p_val = mannwhitneyu(dev_scores, ops_scores, alternative='two-sided')
print(f"Mann-Whitney U Test: U={u_stat}, p-value={p_val:.4f}")

if p_val < 0.05:
    print("Result: Statistically significant difference found.")
else:
    print("Result: No statistically significant difference found.")

# 5. Visualization
plt.figure(figsize=(8, 6))
boxplot_data = [dev_scores.values, ops_scores.values]
plt.boxplot(boxplot_data, labels=['Development', 'Operation'])
plt.title('Governance Score by Lifecycle Stage')
plt.ylabel('Governance Score (0-5)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
