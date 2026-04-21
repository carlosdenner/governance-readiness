# [debug]
import pandas as pd

ds_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(ds_path, low_memory=False)
    
    # Filter for atlas_cases
    atlas_df = df[df['source_table'] == 'atlas_cases']
    
    # Find non-null columns in atlas_df
    non_null_cols = atlas_df.dropna(axis=1, how='all').columns.tolist()
    print(f"Active columns in atlas_cases: {non_null_cols}")
    
    # Check if there is a column related to mitigations in the active columns
    mitigation_cols = [c for c in non_null_cols if 'mitigation' in c.lower() or 'defense' in c.lower()]
    print(f"Potential mitigation columns: {mitigation_cols}")
    
    # Also check step3_mitigation_gaps to see what it links to
    gaps_df = df[df['source_table'] == 'step3_mitigation_gaps']
    print(f"\n--- step3_mitigation_gaps columns: {gaps_df.dropna(axis=1, how='all').columns.tolist()}")
    print(gaps_df.head(3))

except Exception as e:
    print(f"Error: {e}")