import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Load data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
    
    # 2. Map Autonomy Level
    # High: Fully automated or immediate intervention not practicable
    high_autonomy_markers = [
        'Yes - All individual decisions or actions are automated',
        'Other – Immediate human intervention is not practicable'
    ]
    low_autonomy_markers = [
        'No - Some individual decisions or actions require direct human oversight'
    ]
    
    def map_autonomy(val):
        if pd.isna(val):
            return None
        val_str = str(val)
        if any(marker in val_str for marker in high_autonomy_markers):
            return 'High'
        if any(marker in val_str for marker in low_autonomy_markers):
            return 'Low'
        return None

    eo_df['autonomy_level'] = eo_df['57_autonomous_impact'].apply(map_autonomy)

    # 3. Map Appeal Process
    # Yes: Explicitly 'Yes'
    # No: Explicit denials or waivers
    def map_appeal(val):
        if pd.isna(val):
            return None
        val_str = str(val)
        if val_str.strip() == 'Yes':
            return 'Yes'
        if 'No –' in val_str or 'waived' in val_str:
            return 'No'
        return None

    eo_df['has_appeal'] = eo_df['65_appeal_process'].apply(map_appeal)

    # 4. Filter for valid data
    valid_df = eo_df.dropna(subset=['autonomy_level', 'has_appeal'])
    
    print(f"Data points after filtering: {len(valid_df)}")
    
    # 5. Create Contingency Table
    # We want rows to be Autonomy (High/Low) and columns to be Appeal (Yes/No)
    contingency = pd.crosstab(valid_df['autonomy_level'], valid_df['has_appeal'])
    
    # Ensure both rows/cols exist even if counts are 0
    for level in ['High', 'Low']:
        if level not in contingency.index:
            contingency.loc[level] = [0, 0]
    for response in ['Yes', 'No']:
        if response not in contingency.columns:
            contingency[response] = 0
            
    # Reorder for consistency: Rows=[High, Low], Cols=[Yes, No]
    contingency = contingency.loc[['High', 'Low'], ['Yes', 'No']]
    
    print("\n--- Contingency Table (Autonomy vs Appeal) ---")
    print(contingency)
    
    # 6. Statistical Test (Fisher's Exact Test due to small sample sizes)
    odds_ratio, p_value = stats.fisher_exact(contingency)
    
    print("\n--- Statistical Results ---")
    print(f"Fisher's Exact Test p-value: {p_value:.4f}")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    
    # Calculate percentages for plotting
    rates = contingency.div(contingency.sum(axis=1), axis=0)['Yes'] * 100
    
    print("\n--- Appeal Process Rates ---")
    print(rates)

    # 7. Visualization
    plt.figure(figsize=(8, 6))
    bars = plt.bar(rates.index, rates.values, color=['#d62728', '#1f77b4'], alpha=0.7)
    plt.title('Availability of Appeal Process by System Autonomy Level')
    plt.ylabel('Percentage with Formal Appeal Process (%)')
    plt.xlabel('Autonomy Level')
    plt.ylim(0, 100)
    
    # Add counts to bars
    for bar, label in zip(bars, rates.index):
        height = bar.get_height()
        n_total = contingency.loc[label].sum()
        n_yes = contingency.loc[label, 'Yes']
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'n={n_yes}/{n_total}\n({height:.1f}%)',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
