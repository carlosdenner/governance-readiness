import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def analyze_public_sector_intentionality():
    # --- Load Dataset ---
    # Trying both current and parent directory to be safe, though previous run confirmed current.
    filename = 'astalabs_discovery_all_data.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename, low_memory=False)
    elif os.path.exists(os.path.join('..', filename)):
        df = pd.read_csv(os.path.join('..', filename), low_memory=False)
    else:
        print(f"Error: {filename} not found.")
        return

    # --- Filter AIID Data ---
    df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents loaded: {len(df_aiid)}")

    # --- Data Cleaning Logic ---
    col_public = 'Public Sector Deployment'
    col_intent = 'Intentional Harm'

    # Cleaning function for Public Sector
    def clean_sector(val):
        if pd.isna(val):
            return np.nan
        s = str(val).lower().strip()
        if s == 'yes':
            return 'Public'
        elif s == 'no':
            return 'Private'
        return np.nan

    # Cleaning function for Intentional Harm (handling verbose strings)
    def clean_intent(val):
        if pd.isna(val):
            return np.nan
        s = str(val).lower().strip()
        # Check starts with logic based on previous debug output
        if s.startswith('yes'):
            return 'Intentional'
        elif s.startswith('no'):
            return 'Accidental'
        return np.nan

    # Apply cleaning
    df_aiid['Sector_Clean'] = df_aiid[col_public].apply(clean_sector)
    df_aiid['Intent_Clean'] = df_aiid[col_intent].apply(clean_intent)

    # Filter valid rows
    df_clean = df_aiid.dropna(subset=['Sector_Clean', 'Intent_Clean']).copy()
    print(f"Valid rows for analysis: {len(df_clean)}")

    if len(df_clean) < 5:
        print("Insufficient data for analysis.")
        return

    # --- Statistical Analysis ---
    # Contingency Table
    ct = pd.crosstab(df_clean['Sector_Clean'], df_clean['Intent_Clean'])
    print("\n--- Contingency Table (Counts) ---")
    print(ct)
    
    # Proportions
    ct_prop = pd.crosstab(df_clean['Sector_Clean'], df_clean['Intent_Clean'], normalize='index')
    print("\n--- Contingency Table (Proportions) ---")
    print(ct_prop)
    
    # Chi-square Test
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Statistically Significant (Reject H0)")
        print("There IS a significant difference in intentionality between Public and Private sectors.")
    else:
        print("Result: Not Significant (Fail to Reject H0)")
        print("There is NO significant difference in intentionality between Public and Private sectors.")

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ct_prop.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])
    
    ax.set_title('Intentional vs Accidental Harm by Sector (AIID)')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Sector')
    ax.set_ylim(0, 1.0)
    
    # Add labels
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1%', label_type='center', color='white', weight='bold')
    
    plt.legend(title='Harm Intent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_public_sector_intentionality()