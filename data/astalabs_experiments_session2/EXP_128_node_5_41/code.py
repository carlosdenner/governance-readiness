import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
file_name = 'astalabs_discovery_all_data.csv'
if os.path.exists(file_name):
    file_path = file_name
elif os.path.exists(f'../{file_name}'):
    file_path = f'../{file_name}'
else:
    print("Dataset not found.")
    exit(1)

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path, low_memory=False)
    
    # Filter for the relevant source table
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    
    col_policy = '28_iqa_compliance'
    col_practice = '56_monitor_postdeploy'
    
    # --- MAPPING FUNCTIONS ---
    
    def map_iqa_compliance(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).lower()
        
        # Priority: Check for explicit "No" indicators first
        no_keywords = ['not applicable', 'non-public', 'proof of concept', "doesn't appear to meet", 'research']
        if any(k in val_str for k in no_keywords):
            return 'No'
            
        # Check for "Yes" indicators
        yes_keywords = ['policy', 'policies', 'compliance', 'practices', 'checks', 'standard', 'guidance', 'procedures']
        if any(k in val_str for k in yes_keywords):
            return 'Yes'
        
        return np.nan

    def map_monitoring(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).lower()
        
        # Priority: Check for explicit "No" indicators first
        no_keywords = ['no monitoring', 'not available', 'not safety', 'not rights-impacting', 'under development']
        if any(k in val_str for k in no_keywords):
            return 'No'
            
        # Check for "Yes" indicators
        yes_keywords = ['intermittent', 'automated', 'established process', 'manually updated', 'regularly scheduled', 'plan for monitoring']
        if any(k in val_str for k in yes_keywords):
            return 'Yes'
            
        return np.nan

    # Apply mappings
    df_eo['IQA_Mapped'] = df_eo[col_policy].apply(map_iqa_compliance)
    df_eo['Monitor_Mapped'] = df_eo[col_practice].apply(map_monitoring)
    
    # Drop rows where either value couldn't be mapped
    df_clean = df_eo.dropna(subset=['IQA_Mapped', 'Monitor_Mapped']).copy()
    
    print(f"\nTotal rows in source: {len(df_eo)}")
    print(f"Rows with valid mapped data: {len(df_clean)}")
    
    # Create Contingency Table
    contingency_table = pd.crosstab(
        df_clean['IQA_Mapped'], 
        df_clean['Monitor_Mapped']
    )
    
    print("\n--- Contingency Table (Mapped) ---")
    print(contingency_table)
    
    if contingency_table.empty:
        print("No valid data for analysis.")
    else:
        # Ensure 2x2 shape if possible by reindexing
        contingency_table = contingency_table.reindex(index=['Yes', 'No'], columns=['Yes', 'No']).fillna(0)
        
        print("\n--- Reindexed Contingency Table ---")
        print(contingency_table)

        # Chi-Square Test
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Phi Coefficient
        n = contingency_table.sum().sum()
        phi = np.sqrt(chi2 / n) if n > 0 else 0
        
        print("\n--- Statistical Results ---")
        print(f"N: {n}")
        print(f"Chi-Square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4e}")
        print(f"Phi Coefficient: {phi:.4f}")
        
        # Interpretations
        print("\n--- Analysis ---")
        if p < 0.05:
            print("Result: Statistically Significant Association (Reject Null Hypothesis)")
        else:
            print("Result: No Significant Association (Fail to Reject Null Hypothesis)")
            
        # Paper Tiger Analysis (IQA=Yes, Monitor=No)
        iqa_yes_total = contingency_table.loc['Yes'].sum()
        paper_tiger_count = contingency_table.loc['Yes', 'No']
        
        if iqa_yes_total > 0:
            paper_tiger_rate = (paper_tiger_count / iqa_yes_total) * 100
            print(f"\n'Paper Tiger' Rate: {paper_tiger_rate:.1f}% ({int(paper_tiger_count)}/{int(iqa_yes_total)}) of IQA-compliant systems lack operational monitoring.")
        else:
            print("\nNo IQA-compliant systems found to calculate 'Paper Tiger' rate.")
        
        # Visualization
        plt.figure(figsize=(7, 6))
        sns.heatmap(contingency_table, annot=True, fmt='.0f', cmap='Oranges', cbar=False)
        plt.title('Policy-Practice Decoupling:\nIQA Compliance vs. Post-Deployment Monitoring')
        plt.ylabel('IQA Compliance (Policy)')
        plt.xlabel('Post-Deployment Monitoring (Practice)')
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"An error occurred: {e}")