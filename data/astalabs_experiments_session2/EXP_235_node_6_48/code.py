import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def run_experiment():
    # Load dataset
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

    # Filter for AIID incidents
    aiid = df[df['source_table'] == 'aiid_incidents'].copy()

    # --- Helper Functions ---
    def classify_sector(x):
        if pd.isna(x):
            return None
        s = str(x).lower()
        # Healthcare keywords
        if any(k in s for k in ['health', 'medic', 'hospital', 'patient', 'doctor']):
            return 'Healthcare'
        # Law Enforcement keywords
        if any(k in s for k in ['law enforcement', 'police', 'surveillance', 'arrest', 'prison', 'jail']):
            return 'Law Enforcement'
        return None

    def classify_intent(x):
        if pd.isna(x):
            return None
        s = str(x).lower().strip()
        
        # Check for Unintentional first (fixes previous bug where 'unintentional' matched 'intentional')
        # Also check for explicit 'No' which likely means 'Not Intentional'
        if 'unintentional' in s or s == 'no' or 'accidental' in s:
            return 'Unintentional'
        
        # Check for Intentional
        if 'intentional' in s or s == 'yes':
            return 'Intentional'
            
        return None # Exclude ambiguous entries

    # --- Apply Classifications ---
    aiid['target_sector'] = aiid['Sector of Deployment'].apply(classify_sector)
    aiid['harm_intent'] = aiid['Intentional Harm'].apply(classify_intent)

    # Filter dataset
    # We only want rows where both Sector and Intent were successfully classified
    subset = aiid.dropna(subset=['target_sector', 'harm_intent']).copy()

    print(f"Total incidents analyzed after filtering: {len(subset)}")
    
    # Check if we have data
    if len(subset) == 0:
        print("No data found matching criteria.")
        return

    # --- Statistical Analysis ---
    # Generate Contingency Table
    contingency_table = pd.crosstab(subset['target_sector'], subset['harm_intent'])
    print("\nContingency Table (Sector vs. Intentionality):")
    print(contingency_table)

    # Ensure we have enough data dimensions for Chi-Square
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        print("\nInsufficient dimensions for Chi-Square test (need 2x2).")
    else:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nChi-Square Statistic: {chi2:.4f}")
        print(f"P-Value: {p:.4e}")

    # --- Visualization ---
    # Reorder columns to ensure consistent color mapping (Intentional first or second doesn't matter, but must be consistent)
    desired_order = ['Intentional', 'Unintentional']
    # Filter/Sort columns that exist in the data
    cols = [c for c in desired_order if c in contingency_table.columns]
    contingency_table = contingency_table[cols]

    # Normalize rows to 100% for stacked bar chart
    ct_norm = contingency_table.div(contingency_table.sum(axis=1), axis=0)

    # Colors: Red for Intentional, Blue for Unintentional
    color_map = {'Intentional': '#d62728', 'Unintentional': '#1f77b4'}
    colors = [color_map[c] for c in cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    ct_norm.plot(kind='bar', stacked=True, ax=ax, color=colors)

    plt.title('Proportion of Intentional vs. Unintentional Harm by Sector')
    plt.ylabel('Proportion')
    plt.xlabel('Sector')
    plt.xticks(rotation=0)
    plt.legend(title='Harm Intent', loc='upper right', bbox_to_anchor=(1.2, 1))

    # Annotate bars with counts and percentages
    for n, x in enumerate(contingency_table.index):
        row_counts = contingency_table.loc[x]
        row_props = ct_norm.loc[x]
        
        cum_y = 0
        for col in cols:
            count = row_counts[col]
            prop = row_props[col]
            if prop > 0: # Only annotate if segment exists
                # Center text in the segment
                y_pos = cum_y + prop/2
                ax.text(n, y_pos, f"{count}\n({prop:.1%})", 
                        ha='center', va='center', color='white', fontweight='bold')
            cum_y += prop

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()