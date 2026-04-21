import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# [debug] Check current directory and file existence
# print(os.getcwd())
# print(os.listdir('..'))

# Load dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if file is in current directory
    file_path = 'astalabs_discovery_all_data.csv'
    df = pd.read_csv(file_path, low_memory=False)

print("Dataset loaded successfully.")

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
print(f"Filtered for 'atlas_cases': {len(atlas_df)} rows")

# Identify relevant columns for tactics and techniques
# Based on metadata, likely 'tactics' and 'techniques'
# Let's verify columns exist, otherwise search for them
target_cols = ['tactics', 'techniques']
missing_cols = [c for c in target_cols if c not in atlas_df.columns]
if missing_cols:
    print(f"Warning: Columns {missing_cols} not found. searching by keyword...")
    for col in atlas_df.columns:
        if 'tactic' in str(col).lower():
            print(f"Found potential tactic column: {col}")
            atlas_df.rename(columns={col: 'tactics'}, inplace=True)
        if 'technique' in str(col).lower():
            print(f"Found potential technique column: {col}")
            atlas_df.rename(columns={col: 'techniques'}, inplace=True)

# Drop rows with missing tactics or techniques
atlas_df = atlas_df.dropna(subset=['tactics', 'techniques'])
print(f"Rows after dropping nulls in tactics/techniques: {len(atlas_df)}")

# Function to parse lists from strings
def parse_list(s):
    if not isinstance(s, str):
        return []
    # delimiters could be comma or semicolon
    if ';' in s:
        return [x.strip() for x in s.split(';') if x.strip()]
    return [x.strip() for x in s.split(',') if x.strip()]

# Calculate technique counts
atlas_df['technique_list'] = atlas_df['techniques'].apply(parse_list)
atlas_df['technique_count'] = atlas_df['technique_list'].apply(len)

# Identify groups
# Group 1: Cases involving 'Evasion'
# Group 2: Cases involving 'Discovery'

evasion_cases = atlas_df[atlas_df['tactics'].str.contains('Evasion', case=False, na=False)]
discovery_cases = atlas_df[atlas_df['tactics'].str.contains('Discovery', case=False, na=False)]

print(f"\nCases involving 'Evasion': {len(evasion_cases)}")
print(f"Cases involving 'Discovery': {len(discovery_cases)}")

# Check for overlap
overlap_ids = set(evasion_cases.index).intersection(set(discovery_cases.index))
print(f"Overlap (cases in both): {len(overlap_ids)}")

# Prepare data for testing
evasion_counts = evasion_cases['technique_count'].values
discovery_counts = discovery_cases['technique_count'].values

# Descriptive Statistics
print("\n--- Descriptive Statistics (Technique Counts) ---")
print(f"Evasion - Mean: {np.mean(evasion_counts):.2f}, Median: {np.median(evasion_counts):.2f}, Std: {np.std(evasion_counts):.2f}")
print(f"Discovery - Mean: {np.mean(discovery_counts):.2f}, Median: {np.median(discovery_counts):.2f}, Std: {np.std(discovery_counts):.2f}")

# Mann-Whitney U Test
# We use Mann-Whitney because counts are discrete and likely non-normal, and sample sizes are small.
stat, p_value = stats.mannwhitneyu(evasion_counts, discovery_counts, alternative='two-sided')

print("\n--- Hypothesis Test Results (Mann-Whitney U) ---")
print(f"U-statistic: {stat}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant difference in technique counts.")
else:
    print("Result: No statistically significant difference in technique counts.")

# Visualization
plt.figure(figsize=(8, 6))
plt.boxplot([evasion_counts, discovery_counts], labels=['Evasion', 'Discovery'])
plt.title('Distribution of Technique Counts by Adversarial Tactic')
plt.ylabel('Number of Techniques')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()