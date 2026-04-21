import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for relevant source
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# 3. Define Governance Columns (identified in debug phase)
# These represent key controls like Impact Assessment, Independent Eval, etc.
gov_cols = [
    '55_independent_eval', 
    '30_saop_review', 
    '62_disparity_mitigation', 
    '67_opt_out', 
    '52_impact_assessment', 
    '65_appeal_process', 
    '59_ai_notice', 
    '40_has_ato'
]

# 4. Calculate Total Governance Score
def to_binary(val):
    if pd.isna(val):
        return 0
    if isinstance(val, str):
        return 1 if 'yes' in val.lower() else 0
    return 1 if val else 0

eo_df['calculated_gov_score'] = 0
for col in gov_cols:
    eo_df['calculated_gov_score'] += eo_df[col].apply(to_binary)

# 5. Classify Agencies
# We define Defense/Security as agencies involved in national defense, homeland security, justice, or foreign affairs.
defense_keywords = ['Defense', 'Homeland Security', 'Justice', 'State']

def classify_agency(agency_name):
    if pd.isna(agency_name):
        return 'Civilian'
    for kw in defense_keywords:
        if kw in agency_name:
            return 'Defense/Security'
    return 'Civilian'

eo_df['agency_type'] = eo_df['3_agency'].apply(classify_agency)

# Group data
defense_group = eo_df[eo_df['agency_type'] == 'Defense/Security']
civilian_group = eo_df[eo_df['agency_type'] == 'Civilian']

defense_scores = defense_group['calculated_gov_score']
civilian_scores = civilian_group['calculated_gov_score']

# 6. Statistical Analysis
# Descriptive Stats
print(f"Defense/Security Group (N={len(defense_scores)}):")
print(f"  Mean Score: {defense_scores.mean():.4f}")
print(f"  Std Dev:    {defense_scores.std():.4f}")
print(f"  Agencies included: {defense_group['3_agency'].unique()}")

print(f"\nCivilian Group (N={len(civilian_scores)}):")
print(f"  Mean Score: {civilian_scores.mean():.4f}")
print(f"  Std Dev:    {civilian_scores.std():.4f}")

# Levene's Test for Homogeneity of Variance
stat_lev, p_lev = stats.levene(defense_scores, civilian_scores)
print(f"\nLevene's Test: p={p_lev:.4f} (Variances are {'equal' if p_lev > 0.05 else 'unequal'})")

# Independent Samples T-test
t_stat, p_val = stats.ttest_ind(defense_scores, civilian_scores, equal_var=(p_lev > 0.05))
print(f"\nT-test Results:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value:     {p_val:.4e}")

if p_val < 0.05:
    print("  Result: Statistically Significant Difference")
else:
    print("  Result: No Significant Difference")

# 7. Visualization
plt.figure(figsize=(10, 6))
# Use density=True to normalize for different sample sizes
plt.hist(defense_scores, bins=np.arange(0, len(gov_cols) + 2) - 0.5, alpha=0.5, label='Defense/Security', density=True, color='blue', edgecolor='black')
plt.hist(civilian_scores, bins=np.arange(0, len(gov_cols) + 2) - 0.5, alpha=0.5, label='Civilian', density=True, color='orange', edgecolor='black')

plt.xlabel('Governance Readiness Score (Sum of Controls)')
plt.ylabel('Density (Proportion of Agencies)')
plt.title('Comparison of AI Governance Maturity: Defense vs. Civilian Agencies')
plt.xticks(range(0, len(gov_cols) + 1))
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()