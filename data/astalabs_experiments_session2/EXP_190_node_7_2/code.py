import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# Attempt to locate the dataset
file_paths = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
dataset_path = None
for path in file_paths:
    if os.path.exists(path):
        dataset_path = path
        break

if dataset_path is None:
    print("Error: Dataset not found in current or parent directory.")
else:
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)

    # Filter for EO 13960 Scored data
    subset = df[df['source_table'] == 'eo13960_scored'].copy()

    # --- Data Inspection ---
    # Print value counts to understand the distribution for mapping
    print("\n--- '17_impact_type' Value Counts ---")
    print(subset['17_impact_type'].value_counts(dropna=False).head(10))
    
    print("\n--- '38_code_access' Value Counts ---")
    print(subset['38_code_access'].value_counts(dropna=False).head(10))

    # --- Data Processing ---
    
    # 1. Categorize Impact
    # EO 13960 distinguishes between 'Rights-Impacting', 'Safety-Impacting', and 'Other'.
    # Hypothesis focuses on 'High' (Rights/Safety) vs 'Low' (Other/None).
    def map_impact(val):
        val_str = str(val).lower()
        if 'rights' in val_str or 'safety' in val_str or 'high' in val_str:
            return 'High (Rights/Safety)'
        return 'Low/Non-Impacting'

    subset['Impact_Level'] = subset['17_impact_type'].apply(map_impact)

    # 2. Categorize Code Access
    # Identify if code is Open/Publicly available vs Closed.
    def map_code_access(val):
        val_str = str(val).lower()
        # Keywords for open access
        if 'open' in val_str or 'public' in val_str or 'github' in val_str or 'available' in val_str or 'yes' in val_str:
             # exclude explicit 'no' if it appears in 'not available' contexts, but 'available' usually covers it.
             # Let's be stricter: if it says 'no' or 'none' or 'restricted', it's closed.
             if 'no ' in val_str or val_str == 'no' or 'none' in val_str or 'restricted' in val_str:
                 return 'Closed'
             return 'Open'
        return 'Closed'

    subset['Code_Access'] = subset['38_code_access'].apply(map_code_access)

    # --- Statistical Analysis ---
    contingency_table = pd.crosstab(subset['Impact_Level'], subset['Code_Access'])
    print("\n--- Contingency Table (Impact vs. Code Access) ---")
    print(contingency_table)

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")

    # --- Visualization ---
    # Calculate percentage of 'Open' access for each group
    # Row-wise normalization
    props = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    
    if 'Open' in props.columns:
        open_rates = props['Open']
    else:
        open_rates = pd.Series([0, 0], index=props.index)

    print("\nOpen Code Access Rates:")
    print(open_rates)

    plt.figure(figsize=(8, 6))
    ax = open_rates.plot(kind='bar', color=['#d62728', '#2ca02c'], alpha=0.8)
    plt.title('Code Transparency by AI System Impact Level')
    plt.ylabel('Proportion with Open Code Access')
    plt.xlabel('Impact Category')
    plt.ylim(0, max(open_rates.max() * 1.2, 0.1))  # Scale y-axis
    plt.xticks(rotation=0)

    # Add labels
    for p_patch in ax.patches:
        height = p_patch.get_height()
        ax.annotate(f'{height:.1%}', 
                    (p_patch.get_x() + p_patch.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    fontsize=10)

    plt.tight_layout()
    plt.show()
