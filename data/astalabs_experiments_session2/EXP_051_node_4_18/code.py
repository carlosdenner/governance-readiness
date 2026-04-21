import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os

# 1. Load Data
print("Loading datasets...")

# Try current directory first, then parent directory if not found
csv_filename = 'astalabs_discovery_all_data.csv'
json_filename = 'context_crosswalk_evidence.json'

if os.path.exists(csv_filename):
    csv_path = csv_filename
elif os.path.exists(f'../{csv_filename}'):
    csv_path = f'../{csv_filename}'
else:
    print(f"Error: {csv_filename} not found.")
    csv_path = None

if os.path.exists(json_filename):
    json_path = json_filename
elif os.path.exists(f'../{json_filename}'):
    json_path = f'../{json_filename}'
else:
    print(f"Error: {json_filename} not found.")
    json_path = None

if csv_path and json_path:
    # Load CSV and filter for step2_crosswalk_matrix
    try:
        df_csv = pd.read_csv(csv_path, low_memory=False)
        # Filter for the specific source table relevant to the hypothesis
        df_matrix = df_csv[df_csv['source_table'] == 'step2_crosswalk_matrix'].copy()
        
        # Select only relevant columns. 'function' might be in a column named 'function' or similar index
        # Based on previous exploration, 'function' is a column name.
        # 'req_id' is also a column.
        if 'function' in df_matrix.columns and 'req_id' in df_matrix.columns:
            df_matrix = df_matrix[['req_id', 'function']]
        else:
            print(f"Columns 'req_id' or 'function' missing in CSV. Available columns: {df_matrix.columns}")
            sys.exit(1)
            
        print(f"Loaded {len(df_matrix)} rows from CSV step2_crosswalk_matrix.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    # Load JSON
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        df_json = pd.DataFrame(json_data)
        print(f"Loaded {len(df_json)} rows from JSON context_crosswalk_evidence.")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    # 2. Preprocessing & Merging
    # Normalize req_id
    df_matrix['req_id'] = df_matrix['req_id'].astype(str).str.strip()
    df_json['req_id'] = df_json['req_id'].astype(str).str.strip()

    # Merge
    merged_df = pd.merge(df_matrix, df_json, on='req_id', how='inner')
    print(f"Merged dataset shape: {merged_df.shape}")

    if merged_df.empty:
        print("Merged dataframe is empty. Check req_id matching.")
        sys.exit(0)

    # 3. Calculate Control Counts
    def count_controls(controls):
        if isinstance(controls, list):
            return len(controls)
        return 0

    merged_df['control_count'] = merged_df['applicable_controls'].apply(count_controls)

    # Clean Function column
    merged_df['function'] = merged_df['function'].astype(str).str.upper().str.strip()
    
    # Filter to NIST functions if possible, or just print unique values found
    print("Found functions:", merged_df['function'].unique())

    # 4. Statistical Test (Kruskal-Wallis)
    # We compare control counts across the different functions
    functions = merged_df['function'].unique()
    groups = [merged_df[merged_df['function'] == f]['control_count'] for f in functions]

    print("\nDescriptive Statistics:")
    print(merged_df.groupby('function')['control_count'].describe())

    if len(groups) > 1:
        stat, p_value = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis Test Results:")
        print(f"Statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Result: Significant difference found.")
        else:
            print("Result: No significant difference found.")
    else:
        print("Not enough groups for statistical testing.")

    # 5. Visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='function', y='control_count', data=merged_df)
    plt.title('Architectural Control Density by NIST Function')
    plt.xlabel('NIST Function')
    plt.ylabel('Number of Mapped Controls')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("Could not verify file paths.")
