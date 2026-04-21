import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def run_experiment():
    # Load dataset
    file_name = 'astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_name, low_memory=False)
    except FileNotFoundError:
        try:
            df = pd.read_csv('../' + file_name, low_memory=False)
        except FileNotFoundError:
            print("Error: Dataset not found.")
            return

    # Filter for AIID incidents
    aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"Loaded {len(aiid_df)} AIID incidents.")

    # Correct Column Names based on previous exploration
    col_sector = 'Sector of Deployment'
    col_harm_domain = 'Harm Domain'
    col_tangible_harm = 'Tangible Harm'

    # 1. Map Sectors to Groups
    # Group A: 'Admin/Finance' (Public Admin, Finance)
    # Group B: 'Health/Transport' (Healthcare, Transportation)
    
    def map_sector(val):
        if pd.isna(val):
            return None
        val_lower = str(val).lower()
        if 'public administration' in val_lower or 'finance' in val_lower or 'financial' in val_lower:
            return 'Admin/Finance'
        elif 'healthcare' in val_lower or 'transportation' in val_lower:
            return 'Health/Transport'
        return None

    aiid_df['Sector_Group'] = aiid_df[col_sector].apply(map_sector)
    
    # Filter for only relevant sectors
    analysis_df = aiid_df.dropna(subset=['Sector_Group']).copy()
    print(f"\nAnalysis Subset: {len(analysis_df)} incidents in target sectors.")
    print(analysis_df['Sector_Group'].value_counts())

    if len(analysis_df) == 0:
        print("No data found for the target sectors.")
        return

    # 2. Create Binary Bias Variable
    # Keywords: 'bias', 'discrimination', 'civil rights'
    keywords = ['bias', 'discrimination', 'civil rights']

    def is_bias_incident(row):
        # Combine text fields for search
        text_content = f"{str(row[col_harm_domain])} {str(row[col_tangible_harm])}".lower()
        return 1 if any(k in text_content for k in keywords) else 0

    analysis_df['Is_Bias'] = analysis_df.apply(is_bias_incident, axis=1)

    # 3. Statistical Analysis
    # Contingency Table
    contingency = pd.crosstab(analysis_df['Sector_Group'], analysis_df['Is_Bias'])
    print("\nContingency Table (0=No Bias, 1=Bias):")
    print(contingency)

    # Calculate Rates
    stats = analysis_df.groupby('Sector_Group')['Is_Bias'].agg(['count', 'sum', 'mean'])
    stats.columns = ['Total', 'Bias_Count', 'Bias_Rate']
    print("\nDescriptive Statistics:")
    print(stats)

    # Chi-square Test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square Test Results:")
    print(f"Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    if p < 0.05:
        print("Result: Significant difference found (Reject Null).")
    else:
        print("Result: No significant difference found (Fail to Reject Null).")

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    
    # Colors: Admin/Finance (Blue), Health/Transport (Orange)
    colors = ['#1f77b4' if 'Admin' in idx else '#ff7f0e' for idx in stats.index]
    
    bars = plt.bar(stats.index, stats['Bias_Rate'], color=colors, alpha=0.8)
    
    plt.title('Bias/Discrimination Incident Rate by Sector Group')
    plt.ylabel('Proportion of Incidents Involving Bias')
    plt.ylim(0, max(stats['Bias_Rate'].max() * 1.2, 0.1))

    # Annotate bars
    for bar, total, bias_c in zip(bars, stats['Total'], stats['Bias_Count']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.1%} (n={bias_c}/{total})',
                 ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()