import pandas as pd
import scipy.stats as stats
import sys
import os

# Define path handling for dataset
file_name = 'astalabs_discovery_all_data.csv'
possible_paths = [f'../{file_name}', file_name]
dataset_path = next((p for p in possible_paths if os.path.exists(p)), None)

if not dataset_path:
    print(f"Error: Dataset {file_name} not found.")
    sys.exit(1)

print(f"Loading dataset from {dataset_path}...")
try:
    df = pd.read_csv(dataset_path, low_memory=False)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
print(f"ATLAS cases found: {len(atlas_df)}")

# Identify the 'tactics' column
# Metadata indicates column 92 is 'tactics', but sparse CSVs might have shifted names or indices.
tactics_col = 'tactics'
if tactics_col not in atlas_df.columns:
    # search for it
    candidates = [c for c in atlas_df.columns if 'tactics' in str(c).lower()]
    if candidates:
        tactics_col = candidates[0]
        print(f"Found tactics column: {tactics_col}")
    else:
        print("Error: 'tactics' column not found in dataset.")
        print("Available columns:", atlas_df.columns.tolist())
        sys.exit(1)

# Process tactics
# Tactics are likely comma-separated strings. We check for substring presence.
atlas_df[tactics_col] = atlas_df[tactics_col].fillna('').astype(str)

# Create binary flags
# Using case-insensitive match just in case, though metadata suggests capitalized Proper Nouns
atlas_df['has_recon'] = atlas_df[tactics_col].str.contains('Reconnaissance', case=False, regex=False)
atlas_df['has_res_dev'] = atlas_df[tactics_col].str.contains('Resource Development', case=False, regex=False)

# Generate Contingency Table
contingency_table = pd.crosstab(
    atlas_df['has_recon'], 
    atlas_df['has_res_dev'], 
    rownames=['Has Reconnaissance'], 
    colnames=['Has Resource Dev']
)

print("\n--- Contingency Table ---")
print(contingency_table)

# Ensure 2x2 shape for valid output interpretation, fill missing if necessary
# (crosstab might define fewer rows/cols if all are True or all are False)
# We manually construct the 2x2 array for the test to ensure alignment
n_recon_no_res = len(atlas_df[(atlas_df['has_recon'] == True) & (atlas_df['has_res_dev'] == False)])
n_recon_yes_res = len(atlas_df[(atlas_df['has_recon'] == True) & (atlas_df['has_res_dev'] == True)])
n_no_recon_no_res = len(atlas_df[(atlas_df['has_recon'] == False) & (atlas_df['has_res_dev'] == False)])
n_no_recon_yes_res = len(atlas_df[(atlas_df['has_recon'] == False) & (atlas_df['has_res_dev'] == True)])

obs = [[n_no_recon_no_res, n_no_recon_yes_res], [n_recon_no_res, n_recon_yes_res]]
print(f"\nFormatted for Fisher's Test ([[NoRecon/NoRes, NoRecon/YesRes], [Recon/NoRes, Recon/YesRes]]):\n{obs}")

# Perform Fisher's Exact Test
# H0: The presence of Reconnaissance is independent of the presence of Resource Development
odds_ratio, p_value = stats.fisher_exact(obs)

print("\n--- Statistical Test Results ---")
print(f"Fisher's Exact Test p-value: {p_value:.4f}")
print(f"Odds Ratio: {odds_ratio:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Result: Statistically Significant (Reject H0)")
    print("Interpretation: There is a significant association between Reconnaissance and Resource Development tactics.")
else:
    print("Result: Not Statistically Significant (Fail to Reject H0)")
    print("Interpretation: No significant association found between these tactics in this dataset.")
