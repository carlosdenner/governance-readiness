import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def run_experiment():
    # Load dataset
    file_path = '../astalabs_discovery_all_data.csv'
    if not os.path.exists(file_path):
        file_path = 'astalabs_discovery_all_data.csv'
    
    print(f"Loading dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        return

    # Filter for eo13960_scored
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Subset 'eo13960_scored' loaded. Rows: {len(df_eo)}")

    # Columns
    col_svc = '26_public_service'
    col_info = '27_public_info'
    col_opt = '67_opt_out'

    # Robust Cleaning Functions
    def is_public_indicator(val):
        # Logic: If it contains descriptive text (longer than 'No'), it's likely a public use case description.
        # If it is 'No', 'N/A', or empty, it is not.
        if pd.isna(val):
            return False
        s = str(val).strip()
        if not s:
            return False
        s_lower = s.lower()
        if s_lower in ['no', 'n/a', 'none', 'false', '0']:
            return False
        # If it's a long string description (e.g., 'Enabling trusted travelers...'), it implies Yes.
        return True

    def parse_opt_out(val):
        if pd.isna(val):
            return None
        s = str(val).strip().lower()
        if s.startswith('yes'):
            return 'Yes'
        if s.startswith('no') or s.startswith('n/a') or 'waived' in s:
            return 'No'
        return None # 'Other' or ambiguous

    # Apply cleaning
    df_eo['is_public_svc'] = df_eo[col_svc].apply(is_public_indicator)
    df_eo['is_public_info'] = df_eo[col_info].apply(is_public_indicator)
    
    # Define Visibility
    # Public if either service description or info description is present
    df_eo['visibility'] = np.where(df_eo['is_public_svc'] | df_eo['is_public_info'], 'Public', 'Internal')

    # Parse Opt-Out
    df_eo['has_opt_out'] = df_eo[col_opt].apply(parse_opt_out)

    # Filter valid data
    df_clean = df_eo.dropna(subset=['has_opt_out'])
    print(f"Rows with valid Opt-Out status: {len(df_clean)}")
    
    # Contingency Table
    ct = pd.crosstab(df_clean['visibility'], df_clean['has_opt_out'])
    print("\nContingency Table:")
    print(ct)

    if ct.empty or ct.shape != (2, 2):
        print("Warning: Contingency table is not 2x2. Check data distribution.")
        # If 2x2 is not formed, fill missing cols/rows with 0 for robust plotting if possible
        for v in ['Public', 'Internal']:
            if v not in ct.index: ct.loc[v] = [0, 0]
        for c in ['No', 'Yes']:
            if c not in ct.columns: ct[c] = 0
        ct = ct.loc[['Internal', 'Public'], ['No', 'Yes']]
        print("Adjusted Table:")
        print(ct)

    # Statistics
    chi2, p, dof, ex = stats.chi2_contingency(ct)
    print(f"\nChi-Square: {chi2:.4f}, p-value: {p:.4e}")

    # Calculate Opt-Out Rates
    try:
        rate_public = ct.loc['Public', 'Yes'] / ct.loc['Public'].sum()
        rate_internal = ct.loc['Internal', 'Yes'] / ct.loc['Internal'].sum()
        print(f"Opt-Out Rate (Public): {rate_public:.2%}")
        print(f"Opt-Out Rate (Internal): {rate_internal:.2%}")
        
        rr = rate_public / rate_internal if rate_internal > 0 else float('inf')
        print(f"Relative Risk (Public/Internal): {rr:.2f}x")
    except Exception as e:
        print(f"Could not calculate rates: {e}")

    # Plot
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ax = ct_pct.plot(kind='barh', stacked=True, color=['#d9534f', '#5cb85c'], figsize=(10, 6))
    
    plt.title('Opt-Out Implementation by System Visibility')
    plt.xlabel('Percentage')
    plt.ylabel('Visibility')
    plt.axvline(50, color='gray', linestyle='--', alpha=0.5)
    
    # Annotate
    for c in ax.containers:
        # format labels, skip if 0
        labels = [f'{v:.1f}%' if v > 0 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold')
    
    plt.legend(title='Has Opt-Out?', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()