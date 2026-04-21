import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

# [debug]
print("Starting experiment: Commercial vs Custom AI Governance Scores (Retry)")

# 1. Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# 2. Filter for 'eo13960_scored' subset
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded EO 13960 data with {len(eo_data)} rows.")

# 3. Create Grouping Variable based on '22_dev_method'
# Mapping dictionary
dev_method_map = {
    'Developed with contracting resources.': 'Commercial (COTS)',
    'Developed in-house.': 'Custom (GOTS)'
}

# Clean column and map
eo_data['acquisition_type'] = eo_data['22_dev_method'].str.strip().map(dev_method_map)

# Filter for only the two groups of interest
eo_subset = eo_data.dropna(subset=['acquisition_type']).copy()
print(f"Data filtered for Commercial vs Custom comparison: {len(eo_subset)} rows")
print(eo_subset['acquisition_type'].value_counts())

# 4. Implement Text-Scoring Functions for Tier 2 columns

def score_impact_assessment(text):
    # 52_impact_assessment: Score 1 if strictly 'Yes' (ignoring case/whitespace)
    if pd.isna(text): return 0
    t = str(text).lower().strip()
    return 1 if t == 'yes' else 0

def score_disparity_mitigation(text):
    # 62_disparity_mitigation: Length > 10, keywords, not 'None/N/A'
    if pd.isna(text): return 0
    t = str(text).lower().strip()
    if len(t) < 10: return 0
    
    # Keywords indicating real content
    keywords = ['test', 'eval', 'monitor', 'bias', 'fairness', 'mitigat']
    has_keyword = any(k in t for k in keywords)
    
    # Exclusion keywords
    exclusions = ['none', 'n/a', 'waived', 'not applicable']
    is_excluded = any(e in t for e in exclusions)
    
    return 1 if (has_keyword and not is_excluded) else 0

def score_independent_eval(text):
    # 55_independent_eval: Contains 'Yes'
    if pd.isna(text): return 0
    t = str(text).lower().strip()
    return 1 if 'yes' in t else 0

def score_real_world_testing(text):
    # 53_real_world_testing: Contains 'operational environment'
    if pd.isna(text): return 0
    t = str(text).lower().strip()
    return 1 if 'operational environment' in t else 0

def score_postdeploy_monitoring(text):
    # 56_monitor_postdeploy: Monitor/Automated/Established AND NOT 'No monitoring'
    if pd.isna(text): return 0
    t = str(text).lower().strip()
    
    positive_signals = ['monitor', 'automated', 'established process']
    has_signal = any(s in t for s in positive_signals)
    
    negative_signals = ['no monitoring', 'not available']
    has_negative = any(n in t for n in negative_signals)
    
    return 1 if (has_signal and not has_negative) else 0

# Apply scoring
print("Applying text scoring to Tier 2 columns...")
eo_subset['s_52'] = eo_subset['52_impact_assessment'].apply(score_impact_assessment)
eo_subset['s_62'] = eo_subset['62_disparity_mitigation'].apply(score_disparity_mitigation)
eo_subset['s_55'] = eo_subset['55_independent_eval'].apply(score_independent_eval)
eo_subset['s_53'] = eo_subset['53_real_world_testing'].apply(score_real_world_testing)
eo_subset['s_56'] = eo_subset['56_monitor_postdeploy'].apply(score_postdeploy_monitoring)

# Sum scores
score_cols = ['s_52', 's_62', 's_55', 's_53', 's_56']
eo_subset['tier2_score'] = eo_subset[score_cols].sum(axis=1)

# 5. Statistical Analysis
commercial_scores = eo_subset[eo_subset['acquisition_type'] == 'Commercial (COTS)']['tier2_score']
custom_scores = eo_subset[eo_subset['acquisition_type'] == 'Custom (GOTS)']['tier2_score']

mean_comm = commercial_scores.mean()
std_comm = commercial_scores.std()
mean_cust = custom_scores.mean()
std_cust = custom_scores.std()

print(f"\n--- Summary Statistics (Tier 2 Score: 0-{len(score_cols)}) ---")
print(f"Commercial AI (n={len(commercial_scores)}): Mean={mean_comm:.2f}, Std={std_comm:.2f}")
print(f"Custom AI     (n={len(custom_scores)}):     Mean={mean_cust:.2f}, Std={std_cust:.2f}")

# Mann-Whitney U Test
u_stat, p_val = stats.mannwhitneyu(commercial_scores, custom_scores, alternative='two-sided')
print(f"\nMann-Whitney U Test Results:\nU-statistic: {u_stat}\nP-value: {p_val:.4e}")

alpha = 0.05
if p_val < alpha:
    print("Result: Significant difference detected.")
    if mean_comm < mean_cust:
        print("Direction: Commercial AI has LOWER governance scores (supporting hypothesis).")
    else:
        print("Direction: Commercial AI has HIGHER governance scores (contradicting hypothesis).")
else:
    print("Result: No significant difference detected.")

# 6. Visualization
plt.figure(figsize=(10, 6))
data_to_plot = [commercial_scores, custom_scores]
labels = [f'Commercial\n(n={len(commercial_scores)})', f'Custom\n(n={len(custom_scores)})']

plt.boxplot(data_to_plot, labels=labels, showmeans=True)
plt.title('Deep Governance (Tier 2) Compliance by Acquisition Type')
plt.ylabel('Governance Score (0-5)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
