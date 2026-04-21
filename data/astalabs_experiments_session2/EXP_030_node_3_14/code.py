import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

# Load dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored dataset
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# 1. Filter for rows where PII is present
pii_positive = ['Yes', 'yes', 'True', 'true', '1']
eo_pii = eo_df[eo_df['29_contains_pii'].astype(str).str.strip().isin(pii_positive)].copy()

print(f"Total records with PII: {len(eo_pii)}")

# 2. Segment into Groups
le_keywords = ['Law Enforcement', 'Justice', 'Security', 'Defense', 'Intelligence', 'Homeland']
svc_keywords = ['Health', 'Benefits', 'Services', 'Transportation', 'Education', 'Energy', 'Environment', 'Labor', 'Commerce', 'Housing', 'Agriculture']

def classify_topic(topic):
    topic_str = str(topic)
    if any(k in topic_str for k in le_keywords):
        return 'Law Enforcement/Security'
    elif any(k in topic_str for k in svc_keywords):
        return 'Benefits/Services'
    else:
        return 'Other'

eo_pii['group'] = eo_pii['8_topic_area'].apply(classify_topic)

# Filter only for the two groups of interest
analysis_df = eo_pii[eo_pii['group'].isin(['Law Enforcement/Security', 'Benefits/Services'])].copy()

print("\nCounts by Group:")
print(analysis_df['group'].value_counts())

# 3. Define Metric: Bypass Rate
# Bypass = 'No' or Missing (NaN)
def is_bypass(val):
    s = str(val).strip().lower()
    if s == 'no' or s == 'nan':
        return 1
    return 0

analysis_df['bypass_flag'] = analysis_df['30_saop_review'].apply(is_bypass)

# 4. Statistical Test
group_stats = analysis_df.groupby('group')['bypass_flag'].agg(['sum', 'count', 'mean'])
# Rename columns for clarity
group_stats.columns = ['bypassed', 'total', 'rate']

print("\nBypass Statistics:")
print(group_stats)

if len(group_stats) == 2:
    # Extract stats for z-test
    # Note: 'Benefits/Services' is likely at index 0, 'Law Enforcement/Security' at index 1 due to sorting
    le_stats = group_stats.loc['Law Enforcement/Security']
    ben_stats = group_stats.loc['Benefits/Services']
    
    print(f"\nLE Bypass Rate: {le_stats['rate']:.2%}")
    print(f"Benefits Bypass Rate: {ben_stats['rate']:.2%}")
    
    # Hypothesis: LE > Benefits
    # Alternative='larger' means prop(group1) > prop(group2)
    # We pass counts and nobs as lists: [count_LE, count_Ben], [nobs_LE, nobs_Ben]
    
    # FIXED: Use 'bypassed' instead of 'sum' since columns were renamed
    count = [le_stats['bypassed'], ben_stats['bypassed']]
    nobs = [le_stats['total'], ben_stats['total']]
    
    stat, pval = proportions_ztest(count, nobs, alternative='larger')
    
    print(f"\nZ-test Statistic (LE > Benefits): {stat:.4f}")
    print(f"P-value: {pval:.4e}")
    
    # Interpretation check
    if pval < 0.05:
        print("Result: Statistically Significant (Reject Null)")
    else:
        print("Result: Not Significant (Fail to Reject Null)")
        
    # 5. Visualization
    plt.figure(figsize=(8, 6))
    colors = ['skyblue', 'salmon']
    # Ensure order matches
    groups = ['Benefits/Services', 'Law Enforcement/Security']
    rates = [ben_stats['rate'], le_stats['rate']]
    
    plt.bar(groups, rates, color=colors, alpha=0.8)
    plt.ylabel('Rate of SAOP Review Bypass (No/Missing)')
    plt.title('Privacy Oversight Evasion: Law Enforcement vs Benefits')
    plt.ylim(0, 1.0)
    for i, v in enumerate(rates):
        plt.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient groups for comparison.")
