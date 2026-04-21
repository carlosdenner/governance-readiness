import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print("--- Unique Values Analysis ---")
print("16_dev_stage:", eo_data['16_dev_stage'].unique())
print("52_impact_assessment:", eo_data['52_impact_assessment'].unique())

# Mapping Logic
# Operational: Use, Maintenance
# Pre-Operational: Development, Acquisition, Pilot
# Note: Adjusting based on actual values found in standard EO13960 datasets

def map_stage(stage):
    if pd.isna(stage):
        return np.nan
    stage = str(stage).lower()
    if 'use' in stage or 'maintain' in stage or 'maintenance' in stage or 'production' in stage:
        return 'Operational'
    elif 'dev' in stage or 'acq' in stage or 'pilot' in stage or 'plan' in stage:
        return 'Pre-Operational'
    else:
        return 'Other'

def map_assessment(val):
    if pd.isna(val):
        return 0
    val = str(val).lower()
    # Assuming 'yes' indicates completion. strict mapping.
    if val.strip() == 'yes':
        return 1
    return 0

# Apply mappings
eo_data['Status'] = eo_data['16_dev_stage'].apply(map_stage)
eo_data['Has_Assessment'] = eo_data['52_impact_assessment'].apply(map_assessment)

# Filter out 'Other' status if necessary, or just focus on Op vs Pre-Op
analysis_df = eo_data[eo_data['Status'].isin(['Operational', 'Pre-Operational'])].copy()

# Contingency Table
contingency_table = pd.crosstab(analysis_df['Status'], analysis_df['Has_Assessment'])
contingency_table.columns = ['No Assessment', 'Has Assessment']

print("\n--- Contingency Table ---")
print(contingency_table)

# Calculate percentages
rates = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
print("\n--- Assessment Rates (% of Stage) ---")
print(rates)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4f}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant Difference (Reject Null Hypothesis)")
else:
    print("Result: No Statistically Significant Difference (Fail to Reject Null Hypothesis)")

# Visualization
plt.figure(figsize=(8, 6))
rates['Has Assessment'].plot(kind='bar', color=['skyblue', 'orange'])
plt.title('AI Impact Assessment Completion Rate by Development Stage')
plt.ylabel('Completion Rate (%)')
plt.xlabel('Development Stage')
plt.ylim(0, 100)
plt.xticks(rotation=0)
for i, v in enumerate(rates['Has Assessment']):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.show()