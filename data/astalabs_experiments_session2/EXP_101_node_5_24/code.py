import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

def run_experiment():
    print("Starting Experiment: Autonomy-Risk Escalation Analysis (Corrected Mappings)...")

    # 1. Load Dataset
    file_path = 'astalabs_discovery_all_data.csv'
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Failed to load csv: {e}")
        return

    # 2. Filter for AIID Incidents
    df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents found: {len(df_aiid)}")

    # 3. Define Mappings based on specific dataset values
    
    # Autonomy Mapping
    # Autonomy1 -> Low
    # Autonomy2, Autonomy3 -> High
    def map_autonomy_corrected(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if s == 'Autonomy1':
            return 'Low Autonomy'
        elif s in ['Autonomy2', 'Autonomy3']:
            return 'High Autonomy'
        return None

    # Harm Mapping
    # 'tangible harm definitively occurred' -> Tangible
    # 'no tangible harm, near-miss, or issue' -> Intangible
    # Others -> None (Excluded)
    def map_harm_corrected(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if s == 'tangible harm definitively occurred':
            return 'Tangible Harm'
        elif s == 'no tangible harm, near-miss, or issue':
            return 'Intangible Harm'
        # Excluding near-misses and risks as per experiment plan
        return None

    # Apply mappings
    # Note: Column names identified in previous step: 'Autonomy Level' and 'Tangible Harm'
    df_aiid['Autonomy_Bin'] = df_aiid['Autonomy Level'].apply(map_autonomy_corrected)
    df_aiid['Harm_Bin'] = df_aiid['Tangible Harm'].apply(map_harm_corrected)

    # 4. Filter Data
    df_clean = df_aiid.dropna(subset=['Autonomy_Bin', 'Harm_Bin'])
    print(f"Records available for analysis after cleaning: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("No records matched the criteria. Dumping sample values for debugging:")
        print("Autonomy:", df_aiid['Autonomy Level'].unique()[:5])
        print("Harm:", df_aiid['Tangible Harm'].unique()[:5])
        return

    # 5. Statistical Analysis (Chi-Square)
    contingency_table = pd.crosstab(df_clean['Autonomy_Bin'], df_clean['Harm_Bin'])
    print("\n--- Contingency Table ---")
    print(contingency_table)

    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Statistically Significant association found (Reject H0)")
    else:
        print("Result: No statistically significant association found (Fail to reject H0)")

    # 6. Visualization
    # Calculate percentages for the stacked bar chart
    row_props = pd.crosstab(df_clean['Autonomy_Bin'], df_clean['Harm_Bin'], normalize='index') * 100
    
    print("\n--- Row Percentages ---")
    print(row_props)

    plt.figure(figsize=(10, 6))
    ax = row_props.plot(kind='bar', stacked=True, color=['#99ccff', '#ff9999'], edgecolor='black')
    
    plt.title('Distribution of Harm Types by AI Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Percentage of Incidents')
    plt.xticks(rotation=0)
    plt.legend(title='Harm Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Annotate bars
    for c in ax.containers:
        # Filter out labels for very small segments to avoid clutter
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()