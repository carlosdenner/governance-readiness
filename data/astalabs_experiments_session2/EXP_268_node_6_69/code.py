import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
import os

# Define path
data_path = 'astalabs_discovery_all_data.csv'
if not os.path.exists(data_path):
    data_path = '../astalabs_discovery_all_data.csv'

print(f"Loading dataset from: {data_path}")

try:
    df = pd.read_csv(data_path, low_memory=False)
except FileNotFoundError:
    print("Error: Dataset not found.")
    exit(1)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 records loaded: {len(eo_df)}")

# Target columns
date_col = '18_date_initiated'
mitigation_col = '62_disparity_mitigation'

# 1. Parse Dates
eo_df['parsed_date'] = pd.to_datetime(eo_df[date_col], errors='coerce')
eo_df = eo_df.dropna(subset=['parsed_date'])
eo_df['init_year'] = eo_df['parsed_date'].dt.year
eo_df['cohort'] = np.where(eo_df['init_year'] > 2020, 'Post-2020 (Post-EO)', 'Pre-2021 (Legacy)')

# 2. Parse Compliance (Text Classification)
def classify_mitigation(text):
    if pd.isna(text):
        return 0
    text = str(text).lower().strip()
    
    # Strong indicators of valid mitigation controls
    positives = [
        'monitor', 'test', 'eval', 'audit', 'review', 'human', 
        'feedback', 'assess', 'check', 'mitigat', 'adjust', 
        'retrain', 'validation', 'guardrail', 'control', 'standard'
    ]
    
    # Check if text contains any positive indicator
    has_positive = any(p in text for p in positives)
    
    # Check for negations that might invalidate the positive or indicate non-compliance
    # e.g., "No analysis", "Not applicable", "No demographic data used"
    is_negated = False
    if text.startswith(('n/a', 'none', 'no ', 'not ', 'waived')):
        # Usually these start the sentence. 
        # But "No issues found after testing" is positive. 
        # "No analysis performed" is negative.
        if "test" not in text and "review" not in text and "monitor" not in text:
            is_negated = True
    
    # Specific phrase exclusions
    if "no analysis" in text or "no demographic" in text or "not using pii" in text:
        is_negated = True
        
    if has_positive and not is_negated:
        return 1
    return 0

print("\nClassifying mitigation text...")
eo_df['is_compliant'] = eo_df[mitigation_col].apply(classify_mitigation)

# Debug: Check classification examples
print("Classification Check (Sample):")
print(eo_df[[mitigation_col, 'is_compliant']].dropna().head(10))
print(f"Total Compliant: {eo_df['is_compliant'].sum()} / {len(eo_df)}")

# 3. Analyze Cohorts
cohort_stats = eo_df.groupby('cohort')['is_compliant'].agg(['count', 'sum', 'mean'])
cohort_stats.columns = ['Total Systems', 'Compliant Systems', 'Compliance Rate']
print("\nCohort Statistics:")
print(cohort_stats)

# 4. Statistical Test (Two-sample Z-test)
# Post-2020 vs Pre-2021
n_post = cohort_stats.loc['Post-2020 (Post-EO)', 'Total Systems']
k_post = cohort_stats.loc['Post-2020 (Post-EO)', 'Compliant Systems']
n_pre = cohort_stats.loc['Pre-2021 (Legacy)', 'Total Systems']
k_pre = cohort_stats.loc['Pre-2021 (Legacy)', 'Compliant Systems']

stat, pval = proportions_ztest([k_post, k_pre], [n_post, n_pre], alternative='larger')

print(f"\nZ-test Results (Post > Pre):")
print(f"Z-statistic: {stat:.4f}")
print(f"P-value: {pval:.4e}")

if pval < 0.05:
    print("Result: Statistically Significant. The Post-EO cohort has a higher compliance rate.")
else:
    print("Result: Not Statistically Significant.")

# 5. Visualization
plt.figure(figsize=(10, 6))
bars = plt.bar(cohort_stats.index, cohort_stats['Compliance Rate'], color=['#1f77b4', '#ff7f0e'])
plt.title('Impact of EO 13960 on Bias Mitigation Compliance')
plt.ylabel('Proportion of Systems with Bias Mitigation')
plt.ylim(0, max(cohort_stats['Compliance Rate']) * 1.2 if max(cohort_stats['Compliance Rate']) > 0 else 1.0)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.1%}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()