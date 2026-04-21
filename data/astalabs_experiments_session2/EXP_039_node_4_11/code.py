import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def run_experiment():
    print("Loading dataset...")
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        try:
            df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
        except:
            print("Dataset not found.")
            return

    # Filter AIID
    aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Rows: {len(aiid)}")
    
    # Clean column names
    aiid.columns = [c.strip() for c in aiid.columns]

    # 1. Map Autonomy
    # Based on debug: Autonomy1, Autonomy2, Autonomy3
    # Mapping: Low = 1, 2; High = 3
    def map_autonomy(val):
        if pd.isna(val): return None
        s = str(val).lower()
        if 'autonomy1' in s or 'autonomy2' in s:
            return 'Low Autonomy'
        if 'autonomy3' in s:
            return 'High Autonomy'
        return None

    aiid['Autonomy_Bin'] = aiid['Autonomy Level'].apply(map_autonomy)

    # 2. Map Harm Severity (Proxy: Realized Tangible Harm vs Others)
    # 'Tangible Harm' values from debug:
    # - 'tangible harm definitively occurred'
    # - 'imminent risk of tangible harm (near miss) did occur'
    # - 'non-imminent risk of tangible harm (an issue) occurred'
    # - 'no tangible harm, near-miss, or issue'
    
    def map_harm(val):
        if pd.isna(val): return None
        s = str(val).lower()
        if 'definitively occurred' in s:
            return 'Realized Harm'
        elif 'risk' in s or 'near-miss' in s or 'issue' in s or 'no tangible harm' in s:
            return 'Potential/No Harm'
        return None

    aiid['Harm_Bin'] = aiid['Tangible Harm'].apply(map_harm)

    # 3. Create Analysis Subset
    df_analysis = aiid.dropna(subset=['Autonomy_Bin', 'Harm_Bin']).copy()
    
    print("\n--- Data for Analysis ---")
    print(f"Total valid rows: {len(df_analysis)}")
    print(df_analysis['Autonomy_Bin'].value_counts())
    print(df_analysis['Harm_Bin'].value_counts())
    
    if len(df_analysis) < 5:
        print("Insufficient data.")
        return

    # 4. Cross-Tabulation
    ct = pd.crosstab(df_analysis['Autonomy_Bin'], df_analysis['Harm_Bin'])
    print("\n--- Contingency Table (Counts) ---")
    print(ct)
    
    # Normalize to get percentages
    ct_norm = pd.crosstab(df_analysis['Autonomy_Bin'], df_analysis['Harm_Bin'], normalize='index') * 100
    print("\n--- Contingency Table (Percentages) ---")
    print(ct_norm)

    # 5. Statistical Test (Chi-Square)
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"\nChi-Square Test Statistic: {chi2:.4f}")
    print(f"P-value: {p:.5f}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Significant correlation between Autonomy Level and Harm Realization.")
    else:
        print("Result: No significant correlation found.")

    # 6. Visualization
    # We want to show the % of Realized Harm for Low vs High Autonomy
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Plotting
    # Reorder columns if necessary to show 'Realized Harm' clearly
    cols = ['Potential/No Harm', 'Realized Harm']
    # Ensure columns exist
    cols = [c for c in cols if c in ct_norm.columns]
    
    ct_norm[cols].plot(kind='bar', stacked=True, color=['#A8DADC', '#E63946'], ax=plt.gca())
    
    plt.title('Severity of Harm Event by Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Percentage of Incidents')
    plt.legend(title='Harm Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()