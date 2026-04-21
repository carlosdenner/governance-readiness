import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

def map_autonomy(val):
    val = str(val).strip().lower()
    if val.startswith('yes'):
        return 'Autonomous'
    if val.startswith('other'):
        # "immediate human intervention is not practicable" implies autonomy
        return 'Autonomous'
    if val.startswith('no -'):
        return 'Human-in-the-Loop'
    return None

def map_monitoring(val):
    val = str(val).strip().lower()
    if 'no monitoring protocols' in val:
        return 'No'
    if 'under development' in val:
        return 'No'
    if 'intermittent' in val or 'automated' in val or 'established process' in val:
        return 'Yes'
    return None

def run_experiment():
    # Attempt to load the dataset
    file_path = 'astalabs_discovery_all_data.csv'
    if not os.path.exists(file_path):
        file_path = '../astalabs_discovery_all_data.csv'
    
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        return

    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    # Filter for 'eo13960_scored'
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 Scored subset shape: {df_eo.shape}")

    # Apply Mappings
    df_eo['autonomy_category'] = df_eo['57_autonomous_impact'].apply(map_autonomy)
    df_eo['monitoring_category'] = df_eo['56_monitor_postdeploy'].apply(map_monitoring)

    # Drop NaNs in relevant columns
    df_analyzable = df_eo.dropna(subset=['autonomy_category', 'monitoring_category']).copy()

    print(f"Analyzable records after mapping: {len(df_analyzable)}")
    
    if len(df_analyzable) == 0:
        print("No valid data found for analysis after mapping.")
        return

    # Create Contingency Table
    # Rows: Autonomy (Autonomous, HITL)
    # Cols: Monitoring (No, Yes)
    crosstab = pd.crosstab(df_analyzable['autonomy_category'], df_analyzable['monitoring_category'])
    
    # Reorder for consistency if possible
    try:
        crosstab = crosstab.reindex(index=['Human-in-the-Loop', 'Autonomous'], columns=['No', 'Yes'])
    except:
        pass # Keep as is if keys missing
        
    print("\nContingency Table (Counts):")
    print(crosstab)

    # Descriptive Stats
    try:
        n_hitl = crosstab.loc['Human-in-the-Loop'].sum() if 'Human-in-the-Loop' in crosstab.index else 0
        k_hitl = crosstab.loc['Human-in-the-Loop', 'Yes'] if 'Human-in-the-Loop' in crosstab.index and 'Yes' in crosstab.columns else 0
        p_hitl = k_hitl / n_hitl if n_hitl > 0 else 0
        
        n_auto = crosstab.loc['Autonomous'].sum() if 'Autonomous' in crosstab.index else 0
        k_auto = crosstab.loc['Autonomous', 'Yes'] if 'Autonomous' in crosstab.index and 'Yes' in crosstab.columns else 0
        p_auto = k_auto / n_auto if n_auto > 0 else 0
        
        print("\n--- Descriptive Statistics ---")
        print(f"Human-in-the-Loop (n={n_hitl}): {k_hitl} monitored ({p_hitl:.2%})")
        print(f"Autonomous        (n={n_auto}): {k_auto} monitored ({p_auto:.2%})")
    except Exception as e:
        print(f"Error calculating stats: {e}")
        return

    # Chi-Square Test
    # We need a valid 2x2 table for this to be meaningful
    # Check for NaNs in the crosstab (reindex might introduce them)
    crosstab_filled = crosstab.fillna(0)
    
    if n_hitl > 0 and n_auto > 0:
        chi2, p_val, dof, expected = stats.chi2_contingency(crosstab_filled)
        print("\n--- Chi-Square Test Results ---")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"p-value: {p_val:.4e}")
        
        significant = p_val < 0.05
        print("\nConclusion:")
        if significant:
            print("Reject Null Hypothesis: Significant association detected.")
            if p_auto > p_hitl:
                print("Direction: Autonomous systems are MORE likely to be monitored.")
            else:
                print("Direction: Autonomous systems are LESS likely to be monitored.")
        else:
            print("Fail to Reject Null Hypothesis: No significant difference.")
            
        # Visualization
        plt.figure(figsize=(10, 6))
        categories = ['Human-in-the-Loop', 'Autonomous']
        proportions = [p_hitl, p_auto]
        
        bars = plt.bar(categories, proportions, color=['#1f77b4', '#d62728'], alpha=0.8)
        
        plt.ylabel('Proportion with Post-Deployment Monitoring')
        plt.title('The Automation Paradox: Monitoring Rates by Autonomy Level')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar, count, total in zip(bars, [k_hitl, k_auto], [n_hitl, n_auto]):
            height = bar.get_height()
            if total > 0:
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                         f"{height:.1%}\n(n={count}/{total})", 
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()
        
    else:
        print("\nInsufficient data for comparison (missing one of the groups).")

if __name__ == "__main__":
    run_experiment()