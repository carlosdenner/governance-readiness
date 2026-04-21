import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def run_experiment():
    print("Starting Experiment: Sectoral Bias Blindspots (Improved Text Analysis)...")
    
    # 1. Load dataset
    filename = 'astalabs_discovery_all_data.csv'
    if not os.path.exists(filename):
        filename = '../astalabs_discovery_all_data.csv'
    
    try:
        df = pd.read_csv(filename, low_memory=False)
        print(f"Dataset loaded from: {filename}")
    except FileNotFoundError:
        print(f"Error: Dataset {filename} not found.")
        return

    # Filter for EO13960 source table
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 scored records: {len(df_eo)}")

    # 2. Segment Data by Topic Area
    def get_sector(s):
        if pd.isna(s): return None
        s_str = str(s).lower()
        if any(x in s_str for x in ['law enforcement', 'justice', 'security']):
            return 'Law Enforcement'
        if 'health' in s_str:
            return 'Health'
        return None

    df_eo['analysis_sector'] = df_eo['8_topic_area'].apply(get_sector)
    
    # Filter for target sectors
    df_subset = df_eo[df_eo['analysis_sector'].isin(['Law Enforcement', 'Health'])].copy()
    
    print("\n--- Data Segmentation ---")
    print(df_subset['analysis_sector'].value_counts())

    if df_subset.empty:
        print("No data found for target sectors.")
        return

    # 3. Improved Classification Logic
    def classify_mitigation(val):
        if pd.isna(val):
            return 0
        text = str(val).strip().lower()
        if not text or text == 'nan':
            return 0
            
        # Keywords indicating affirmative action
        # Added 'human' based on previous output "human can evaluate"
        positive_keywords = [
            'test', 'eval', 'monitor', 'review', 'analy', 'assess', 
            'mitigat', 'audit', 'check', 'ensur', 'bias', 'fair', 
            'equit', 'control', 'human', 'scan', 'detect'
        ]
        
        # Negative starts
        negative_starts = ('no ', 'none', 'n/a', 'not ')
        is_negative_start = text.startswith(negative_starts) or text in ['no', 'none', 'n/a']
        
        has_positive = any(kw in text for kw in positive_keywords)
        
        if is_negative_start:
            # Heuristic: If it starts with negative but is long or contains contrast, it might be a "Soft Negative" (Qualified)
            # E.g. "None, however we..." or "None for X, but Y..."
            if len(text) > 60 and has_positive:
                return 1
            return 0
        else:
            # If it's not explicitly negative at start, check for positive content
            if has_positive:
                return 1
            return 0

    df_subset['mitigation_flag'] = df_subset['62_disparity_mitigation'].apply(classify_mitigation)

    # Validation of classification
    print("\n--- Classification Validation ---")
    print("Sample Positive (1):")
    print(df_subset[df_subset['mitigation_flag']==1]['62_disparity_mitigation'].head(3).tolist())
    print("\nSample Negative (0):")
    print(df_subset[df_subset['mitigation_flag']==0]['62_disparity_mitigation'].head(3).tolist())

    # 4. Comparative Analysis
    summary = df_subset.groupby('analysis_sector')['mitigation_flag'].agg(['count', 'sum', 'mean'])
    summary.columns = ['Total', 'Mitigated', 'Rate']
    
    print("\n--- Summary Statistics ---")
    print(summary)

    # 5. Statistical Test (Chi-Square)
    contingency = pd.crosstab(df_subset['analysis_sector'], df_subset['mitigation_flag'])
    print("\n--- Contingency Table ---")
    print(contingency)

    # Ensure valid shape
    if 0 not in contingency.columns: contingency[0] = 0
    if 1 not in contingency.columns: contingency[1] = 0
    contingency = contingency[[0, 1]] # Ensure order

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")

    if p < 0.05:
        print("Result: Statistically significant difference detected.")
    else:
        print("Result: No statistically significant difference detected.")

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    # Color: Health=Blue, LE=Red
    colors = ['#1f77b4', '#d62728']
    
    ax = summary['Rate'].plot(kind='bar', color=colors, alpha=0.8, edgecolor='black', rot=0)
    
    plt.title('Disparity Mitigation Rates: Health vs. Law Enforcement\n(Broad Keyword Search)')
    plt.ylabel('Proportion of Affirmative Mitigation Evidence')
    plt.xlabel('Sector')
    plt.ylim(0, 1.0) # Keep 0-1 scale for context, or zoom in if values are small but visible
    if summary['Rate'].max() < 0.2:
        plt.ylim(0, 0.25) # Zoom if rates are still low
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(summary['Rate']):
        ax.text(i, v + (plt.ylim()[1]*0.02), f"{v:.1%}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()