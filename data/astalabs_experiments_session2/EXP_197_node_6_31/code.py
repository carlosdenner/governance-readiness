import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

def run_experiment():
    # Load dataset
    file_path = '../astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        try:
            df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        except FileNotFoundError:
            print("Dataset not found.")
            return

    # Filter for EO 13960 scored data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    
    # 1. Process '16_dev_stage'
    # Define mapping
    # Operational: Operation, Maintenance, Use, Deployed
    # Development: Development, Acquisition, Planning, Initiation, Pilot
    def bin_stage(val):
        if pd.isna(val):
            return None
        s = str(val).lower()
        if any(x in s for x in ['oper', 'use', 'maint', 'deploy']):
            return 'Operational'
        elif any(x in s for x in ['dev', 'plan', 'acq', 'init', 'pilot', 'test']):
            return 'Development'
        return None # Exclude retired/unknown

    df_eo['stage_bin'] = df_eo['16_dev_stage'].apply(bin_stage)
    
    # Filter relevant rows
    df_analysis = df_eo.dropna(subset=['stage_bin']).copy()
    
    # 2. Process '56_monitor_postdeploy' (Monitoring Documentation)
    # Heuristic: treat typical negative responses as 0, substantive text as 1
    negatives = ['no', 'none', 'n/a', 'na', 'not applicable', '0', '-', 'false', 'unknown', 'tbd']
    
    def bin_monitoring(val):
        if pd.isna(val):
            return 0
        s = str(val).strip().lower()
        if s in negatives:
            return 0
        # Check for "No ..." sentences that are actually negations
        if s.startswith('no ') and len(s) < 20:
            return 0
        return 1

    df_analysis['monitor_bin'] = df_analysis['56_monitor_postdeploy'].apply(bin_monitoring)
    
    # 3. Contingency Table
    contingency = pd.crosstab(df_analysis['stage_bin'], df_analysis['monitor_bin'])
    # Ensure columns exist (handle case where one might be missing)
    if 0 not in contingency.columns: contingency[0] = 0
    if 1 not in contingency.columns: contingency[1] = 0
    contingency = contingency[[0, 1]]
    contingency.columns = ['No Documentation', 'Has Documentation']
    
    print("--- Contingency Table (Stage vs Monitoring) ---")
    print(contingency)
    print("\n")

    # 4. Summary Statistics
    summary = contingency.copy()
    summary['Total'] = summary['No Documentation'] + summary['Has Documentation']
    summary['% Documented'] = (summary['Has Documentation'] / summary['Total']) * 100
    
    print("--- Summary Statistics ---")
    print(summary[['Total', '% Documented']])
    print("\n")

    # 5. Chi-square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print("--- Chi-Square Test Results ---")
    print(f"Chi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Statistically Significant")
    else:
        print("Result: Not Statistically Significant")

    # 6. Odds Ratio
    # (Operational_Yes * Development_No) / (Operational_No * Development_Yes)
    try:
        op_yes = contingency.loc['Operational', 'Has Documentation']
        op_no = contingency.loc['Operational', 'No Documentation']
        dev_yes = contingency.loc['Development', 'Has Documentation']
        dev_no = contingency.loc['Development', 'No Documentation']
        
        # Handle division by zero
        if op_no == 0 or dev_yes == 0:
             print("Odds Ratio: Undefined (Zero count in denominator)")
        else:
             or_val = (op_yes * dev_no) / (op_no * dev_yes)
             print(f"Odds Ratio (Operational vs Development): {or_val:.4f}")
    except KeyError:
        print("Error calculating odds ratio: Missing category keys.")

    # 7. Visualization
    plt.figure(figsize=(8, 6))
    bars = plt.bar(summary.index, summary['% Documented'], color=['#ff9999', '#66b3ff'])
    plt.title('Documented Monitoring by Lifecycle Stage')
    plt.ylabel('% with Documented Monitoring')
    plt.xlabel('Lifecycle Stage')
    plt.ylim(0, max(summary['% Documented']) * 1.2)
    
    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()