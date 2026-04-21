import pandas as pd
import json
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# 1. Load CSV for 'function' and 'req_id'
csv_path = 'astalabs_discovery_all_data.csv'
if not os.path.exists(csv_path):
    csv_path = f"../{csv_path}"

df_csv = pd.read_csv(csv_path, low_memory=False)

# Filter for crosswalk matrix
csv_subset = df_csv[df_csv['source_table'] == 'step2_crosswalk_matrix'].copy()

# Keep relevant columns
if 'function' not in csv_subset.columns or 'req_id' not in csv_subset.columns:
    print("Error: Missing columns in CSV")
    exit(1)

# Normalize req_id in CSV (strip whitespace)
csv_subset['req_id'] = csv_subset['req_id'].astype(str).str.strip()
csv_data = csv_subset[['req_id', 'function']]

print(f"CSV Matrix Rows: {len(csv_data)}")
print(f"CSV Head req_id: {csv_data['req_id'].head().tolist()}")

# 2. Load JSON for 'req_id' and 'applicable_controls'
json_path = 'context_crosswalk_evidence.json'
if not os.path.exists(json_path):
    json_path = f"../{json_path}"

with open(json_path, 'r') as f:
    json_list = json.load(f)

df_json = pd.DataFrame(json_list)
# Normalize req_id in JSON
df_json['req_id'] = df_json['req_id'].astype(str).str.strip()

print(f"JSON Rows: {len(df_json)}")
print(f"JSON Head req_id: {df_json['req_id'].head().tolist()}")

# 3. Merge
# Try inner join
merged = pd.merge(csv_data, df_json, on='req_id', how='inner')
print(f"Merged Rows: {len(merged)}")

# If merge failed (0 rows) and counts match (42), try positional merge as fallback
if len(merged) == 0 and len(csv_data) == 42 and len(df_json) == 42:
    print("Warning: Merge on req_id failed. Attempting positional merge based on row order.")
    csv_data = csv_data.reset_index(drop=True)
    df_json = df_json.reset_index(drop=True)
    merged = pd.concat([csv_data, df_json.drop(columns=['req_id'])], axis=1)
    print(f"Positional Merge Rows: {len(merged)}")

# 4. Calculate Control Counts
def calc_len(x):
    if isinstance(x, list):
        return len(x)
    return 0

merged['control_count'] = merged['applicable_controls'].apply(calc_len)

# 5. Analyze MAP vs MANAGE
target_functions = ['MAP', 'MANAGE']
analysis_set = merged[merged['function'].isin(target_functions)].copy()

print(f"Analysis Set Size: {len(analysis_set)}")
print(analysis_set.groupby('function')['control_count'].describe())

map_data = analysis_set[analysis_set['function'] == 'MAP']['control_count']
manage_data = analysis_set[analysis_set['function'] == 'MANAGE']['control_count']

if len(map_data) > 0 and len(manage_data) > 0:
    t_stat, p_val = stats.ttest_ind(map_data, manage_data, equal_var=False)
    print(f"\nT-test Results (MAP vs MANAGE):\nT-statistic: {t_stat:.4f}\nP-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Result: Significant difference.")
    else:
        print("Result: No significant difference.")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.boxplot([map_data, manage_data], tick_labels=['MAP', 'MANAGE'])
    plt.title('Technical Controls: MAP vs MANAGE')
    plt.ylabel('Count of Controls')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("Insufficient data for analysis.")
