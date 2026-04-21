# [debug]
import pandas as pd
import os

filename = 'astalabs_discovery_all_data.csv'
if not os.path.exists(filename):
    filename = '../astalabs_discovery_all_data.csv'

print(f"Loading {filename}...")
try:
    df = pd.read_csv(filename, low_memory=False)
    print("Dataset loaded successfully.")
    
    # Check for ATLAS related data
    atlas_df = df[df['source_table'] == 'atlas_cases']
    print(f"ATLAS cases shape: {atlas_df.shape}")
    
    # Check potential technique columns
    tech_cols = [c for c in df.columns if 'technique' in c.lower()]
    print(f"Columns containing 'technique': {tech_cols}")
    
    # Check which rows have techniques
    for col in tech_cols:
        non_null = df[df[col].notna()]
        if not non_null.empty:
            print(f"Column '{col}' has {len(non_null)} non-null values.")
            print(f"Source tables for '{col}': {non_null['source_table'].unique()}")
            print(f"Sample '{col}': {non_null[col].iloc[0]}")

    # Check for case_id
    case_cols = [c for c in df.columns if 'case' in c.lower()]
    print(f"Columns containing 'case': {case_cols}")
    
    # Specifically check if atlas_cases has case_id and if it's numeric or string
    if 'case_id' in atlas_df.columns:
        print("Sample case_id from atlas_cases:", atlas_df['case_id'].head().tolist())
        
except Exception as e:
    print(f"Error: {e}")
