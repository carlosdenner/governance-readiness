import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        exit(1)

# Filter for EO 13960
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# Define Security vs Civilian Agencies
# Security agencies: DOD, DHS, DOJ, State
security_keywords = ['Defense', 'Homeland', 'Justice', 'State']

def classify_agency(agency_name):
    if pd.isna(agency_name):
        return 'Civilian'
    name = str(agency_name)
    if any(keyword in name for keyword in security_keywords):
        return 'Security'
    return 'Civilian'

eo_df['agency_type'] = eo_df['3_agency'].apply(classify_agency)

# Map Notice (59_ai_notice)
# Identify non-compliance indicators to map to 0; others to 1.
negative_notice_starts = [
    'None of the above',
    'N/A',
    'Agency CAIO has waived',
    'AI is not safety'
]

def map_notice(val):
    if pd.isna(val):
        return 0
    s = str(val).strip()
    # If it starts with any negative phrase, it's a 0
    for neg in negative_notice_starts:
        if s.startswith(neg):
            return 0
    # Otherwise, assuming it indicates a method of notice (Online, In-person, Email, etc.)
    return 1

eo_df['notice_score'] = eo_df['59_ai_notice'].apply(map_notice)

# Map Opt-out (67_opt_out)
# Strict mapping: Only 'Yes' is 1.
def map_opt_out(val):
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s.lower() == 'yes':
        return 1
    return 0

eo_df['opt_out_score'] = eo_df['67_opt_out'].apply(map_opt_out)

# Calculate Composite Rights Score (0-2)
eo_df['rights_score'] = eo_df['notice_score'] + eo_df['opt_out_score']

# Separate Groups
security_scores = eo_df[eo_df['agency_type'] == 'Security']['rights_score']
civilian_scores = eo_df[eo_df['agency_type'] == 'Civilian']['rights_score']

# Perform Statistical Test (Mann-Whitney U)
stat, p_val = mannwhitneyu(civilian_scores, security_scores, alternative='two-sided')

# Output Results
print("--- Experiment Results: Security-Rights Divergence ---")
print(f"Security/Defense Agencies (N={len(security_scores)}): Mean Rights Score = {security_scores.mean():.3f}")
print(f"Civilian Agencies (N={len(civilian_scores)}): Mean Rights Score = {civilian_scores.mean():.3f}")
print(f"Mann-Whitney U Test: Statistic={stat}, p-value={p_val:.5f}")

if p_val < 0.05:
    print("Conclusion: Significant difference found. The hypothesis is supported.")
else:
    print("Conclusion: No significant difference found. The hypothesis is rejected.")

# Visualization
data_to_plot = [civilian_scores, security_scores]
plt.figure(figsize=(8, 6))
plt.boxplot(data_to_plot)
plt.xticks([1, 2], ['Civilian', 'Security/Defense'])
plt.title('Rights-Preserving Controls (Notice + Opt-Out) by Agency Type')
plt.ylabel('Rights Score (0-2)')
plt.yticks([0, 1, 2])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()