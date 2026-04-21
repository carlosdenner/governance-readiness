import pandas as pd
import numpy as np
import sys
import subprocess
import os

# Ensure statsmodels is installed
try:
    from statsmodels.stats.proportion import proportions_ztest
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "statsmodels"])
    from statsmodels.stats.proportion import proportions_ztest

def load_data():
    # Try loading from current directory first, then parent directory
    files = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
    for f in files:
        if os.path.exists(f):
            print(f"Loading dataset from {f}...")
            return pd.read_csv(f, low_memory=False)
    print("Error: Dataset not found in current or parent directory.")
    return None

def run_experiment():
    df = load_data()
    if df is None:
        return

    # Filter data sources
    aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    atlas = df[df['source_table'] == 'atlas_cases'].copy()

    print(f"Total AIID Incidents: {len(aiid)}")
    print(f"Total ATLAS Cases: {len(atlas)}")

    # --- Define Sector Normalization Logic ---
    # Keywords for healthcare domain
    health_keywords = ['health', 'medic', 'hospital', 'clinic', 'patient', 'doctor', 'nurse', 'pharm', 'surgery', 'diagnostic', 'biomedical']

    def is_healthcare(text):
        if not isinstance(text, str):
            return False
        text = text.lower()
        return any(k in text for k in health_keywords)

    # --- Process AIID ---
    # Metadata suggests 'Sector of Deployment' for AIID
    # We'll check 'Sector of Deployment' and 'sector'
    aiid_col = 'Sector of Deployment' if 'Sector of Deployment' in aiid.columns and aiid['Sector of Deployment'].notna().any() else 'sector'
    print(f"Using AIID sector column: {aiid_col}")
    
    aiid['is_healthcare'] = aiid[aiid_col].apply(is_healthcare)
    
    # --- Process ATLAS ---
    # Metadata suggests 'sector' for ATLAS
    atlas_col = 'sector' if 'sector' in atlas.columns and atlas['sector'].notna().any() else 'Sector of Deployment'
    # Fallback to checking name/summary if sector is missing (ATLAS is small)
    if atlas_col not in atlas.columns or atlas[atlas_col].isna().all():
        print("Warning: ATLAS sector column missing or empty. Using 'name' and 'summary' for context.")
        atlas['combined_text'] = atlas['name'].fillna('') + ' ' + atlas['summary'].fillna('')
        atlas['is_healthcare'] = atlas['combined_text'].apply(is_healthcare)
    else:
        print(f"Using ATLAS sector column: {atlas_col}")
        atlas['is_healthcare'] = atlas[atlas_col].apply(is_healthcare)

    # --- Calculate Stats ---
    n_aiid = len(aiid)
    k_aiid = aiid['is_healthcare'].sum()
    prop_aiid = k_aiid / n_aiid if n_aiid > 0 else 0

    n_atlas = len(atlas)
    k_atlas = atlas['is_healthcare'].sum()
    prop_atlas = k_atlas / n_atlas if n_atlas > 0 else 0

    print("\n--- Results ---")
    print(f"AIID (Real Incidents): {k_aiid}/{n_aiid} ({prop_aiid:.2%}) classified as Healthcare")
    print(f"ATLAS (Threat Models): {k_atlas}/{n_atlas} ({prop_atlas:.2%}) classified as Healthcare")

    # --- Statistical Test ---
    if n_aiid > 0 and n_atlas > 0:
        count = np.array([k_aiid, k_atlas])
        nobs = np.array([n_aiid, n_atlas])
        
        # Two-sided Z-test
        stat, pval = proportions_ztest(count, nobs)
        print(f"\nTwo-Proportion Z-Test:")
        print(f"Z-score: {stat:.4f}")
        print(f"P-value: {pval:.4f}")
        
        alpha = 0.05
        if pval < alpha:
            print("Result: Statistically Significant Difference")
            if prop_atlas < prop_aiid:
                print("Conclusion: Healthcare is significantly UNDER-represented in ATLAS threat models compared to real-world incidents.")
            else:
                print("Conclusion: Healthcare is significantly OVER-represented in ATLAS threat models compared to real-world incidents.")
        else:
            print("Result: No Statistically Significant Difference")
    else:
        print("Insufficient data for statistical testing.")

if __name__ == "__main__":
    run_experiment()