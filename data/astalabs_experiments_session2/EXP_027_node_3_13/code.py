import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def run_experiment():
    print("Starting Autonomy-Severity analysis (Attempt 2)...")
    
    # 1. Load the dataset
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

    # 2. Filter for AIID incidents
    aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents loaded: {len(aiid)}")

    # 3. Identify correct columns
    # Based on previous exploration, we look for 'Autonomy Level' and 'Tangible Harm'
    # The dataset metadata suggests column names might be close to these.
    cols = aiid.columns
    autonomy_col = next((c for c in cols if 'Autonomy Level' in c), 'Autonomy Level')
    harm_col = next((c for c in cols if 'Tangible Harm' in c), 'Tangible Harm')
    
    print(f"Using columns: '{autonomy_col}' and '{harm_col}'")

    # 4. Map Autonomy Level
    # Previous findings: ['Autonomy1', 'Autonomy2', 'Autonomy3', 'unclear']
    def map_autonomy(val):
        s = str(val).strip()
        if s == 'Autonomy3':
            return 'High'
        elif s in ['Autonomy1', 'Autonomy2']:
            return 'Low'
        else:
            return np.nan  # Exclude 'unclear' or nan

    aiid['Autonomy_Bucket'] = aiid[autonomy_col].apply(map_autonomy)
    
    # 5. Map Tangible Harm
    # Previous findings: ['tangible harm definitively occurred', 'no tangible harm...', ...]
    # Hypothesis: Physical (Tangible) vs Intangible (Non-Physical)
    def map_harm(val):
        s = str(val).lower()
        if 'definitively occurred' in s:
            return 'Physical'  # Tangible harm happened
        elif 'no tangible harm' in s or 'issue' in s or 'risk' in s:
            return 'Intangible' # No tangible harm (economic, reputation, or near-miss)
        else:
            return np.nan # Exclude 'unclear'

    aiid['Harm_Bucket'] = aiid[harm_col].apply(map_harm)

    # 6. Drop NaNs in buckets
    valid_data = aiid.dropna(subset=['Autonomy_Bucket', 'Harm_Bucket'])
    print(f"Valid rows after mapping: {len(valid_data)}")
    
    if len(valid_data) == 0:
        print("No valid data found after mapping. Check values again.")
        print("Autonomy values:", aiid[autonomy_col].unique())
        print("Harm values:", aiid[harm_col].unique())
        return

    # 7. Generate Contingency Table
    ct = pd.crosstab(valid_data['Autonomy_Bucket'], valid_data['Harm_Bucket'])
    print("\nContingency Table (Count):")
    print(ct)

    # 8. Calculate Proportions
    # We want to see % of Physical harm in High vs Low autonomy
    props = pd.crosstab(valid_data['Autonomy_Bucket'], valid_data['Harm_Bucket'], normalize='index') * 100
    print("\nProportions (%):")
    print(props.round(2))

    # 9. Statistical Test (Chi-Square)
    if ct.shape == (2, 2):
        chi2, p, dof, ex = stats.chi2_contingency(ct)
        print(f"\nChi-square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4e}")
        
        alpha = 0.05
        if p < alpha:
            print("Result: Significant association found (p < 0.05).")
        else:
            print("Result: No significant association found (p >= 0.05).")
    else:
        print("\nContingency table is not 2x2, skipping Chi-square.")

    # 10. Visualization
    # Stacked bar chart
    ax = props.plot(kind='bar', stacked=True, color=['lightgray', 'firebrick'], figsize=(8, 6))
    plt.title('Proportion of Physical vs Intangible Harm by Autonomy Level')
    plt.ylabel('Percentage')
    plt.xlabel('Autonomy Level')
    plt.xticks(rotation=0)
    
    # Annotate bars
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
        
    plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()