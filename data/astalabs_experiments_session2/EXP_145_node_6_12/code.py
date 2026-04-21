import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Loaded EO 13960 data with {len(eo_df)} records.")

# --- Step 1: Normalize Scoring Columns ---
# Columns identified: '38_code_access', '34_data_docs', '56_monitor_postdeploy', '31_data_catalog'
# Inspect unique values to determine mapping logic
score_cols = ['38_code_access', '34_data_docs', '56_monitor_postdeploy', '31_data_catalog']

print("\n--- Unique Values in Scoring Columns (Pre-normalization) ---")
for col in score_cols:
    if col in eo_df.columns:
        print(f"{col}: {eo_df[col].unique()}")
    else:
        print(f"{col}: MISSING")

# Function to map values to binary
def normalize_to_binary(val):
    if pd.isna(val):
        return 0
    s = str(val).lower()
    # Common affirmative values in this dataset based on previous context
    if any(x in s for x in ['yes', 'true', '1', 'open', 'public']):
        return 1
    return 0

# Apply normalization
for col in score_cols:
    if col in eo_df.columns:
        eo_df[col + '_score'] = eo_df[col].apply(normalize_to_binary)
    else:
        eo_df[col + '_score'] = 0

# Calculate Composite Integration Score
eo_df['integration_score'] = eo_df[[c + '_score' for c in score_cols]].sum(axis=1)

print("\n--- Composite Score Stats ---")
print(eo_df['integration_score'].describe())

# --- Step 2: Define Agency Clusters ---
# Inspect Agencies to ensure correct keyword matching
print("\n--- Available Agencies (Top 20) ---")
print(eo_df['3_agency'].value_counts().head(20))

def map_agency_cluster(agency_name):
    if pd.isna(agency_name):
        return None
    agency = str(agency_name).lower()
    
    # Scientific Cluster
    scientific_keywords = [
        'aeronautics', 'nasa', 
        'science foundation', 'nsf', 
        'energy', 
        'commerce', 
        'nist'
    ]
    if any(k in agency for k in scientific_keywords):
        return 'Scientific'
    
    # Benefits Cluster
    benefits_keywords = [
        'social security', 'ssa',
        'veterans', 'va',
        'housing', 'hud',
        'education'
    ]
    if any(k in agency for k in benefits_keywords):
        return 'Benefits'
    
    return 'Other'

eo_df['cluster'] = eo_df['3_agency'].apply(map_agency_cluster)

# Filter for target clusters
analysis_df = eo_df[eo_df['cluster'].isin(['Scientific', 'Benefits'])].copy()

print("\n--- Cluster Counts ---")
print(analysis_df['cluster'].value_counts())

# --- Step 3: Statistical Test ---
sci_scores = analysis_df[analysis_df['cluster'] == 'Scientific']['integration_score']
ben_scores = analysis_df[analysis_df['cluster'] == 'Benefits']['integration_score']

# Independent Samples T-test
t_stat, p_val = stats.ttest_ind(sci_scores, ben_scores, equal_var=False)

print("\n--- T-Test Results ---")
print(f"Scientific Mean: {sci_scores.mean():.2f} (n={len(sci_scores)})")
print(f"Benefits Mean:   {ben_scores.mean():.2f} (n={len(ben_scores)})")
print(f"T-statistic:     {t_stat:.4f}")
print(f"P-value:         {p_val:.4e}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically Significant Difference")
else:
    print("Result: No Significant Difference")

# --- Step 4: Visualization ---
plt.figure(figsize=(8, 6))
data_to_plot = [sci_scores, ben_scores]
plt.boxplot(data_to_plot, labels=['Scientific Agencies', 'Benefits Agencies'])
plt.title('Integration Readiness Scores by Agency Type')
plt.ylabel('Composite Score (0-4)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()