import pandas as pd
import scipy.stats as stats
import os

def run_experiment():
    # Attempt to locate the dataset in current or parent directory
    filename = 'astalabs_discovery_all_data.csv'
    paths = [filename, f'../{filename}']
    df = None
    for p in paths:
        if os.path.exists(p):
            print(f"Dataset found at: {p}")
            df = pd.read_csv(p, low_memory=False)
            break
            
    if df is None:
        print("Error: Dataset not found.")
        return

    # Filter for EO 13960 source
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Total EO 13960 records: {len(df_eo)}")

    # Map '16_dev_stage' to Lifecycle Phases
    def map_stage(val):
        s = str(val).lower()
        # Operational keywords
        if any(x in s for x in ['operation', 'maintenance', 'use', 'production', 'sustainment']):
            return 'Operational'
        # Pre-Operational keywords
        if any(x in s for x in ['development', 'acquisition', 'planning', 'design', 'pilot', 'test']):
            return 'Pre-Operational'
        return None

    df_eo['Phase'] = df_eo['16_dev_stage'].apply(map_stage)
    df_analysis = df_eo.dropna(subset=['Phase']).copy()
    print(f"Records with valid phase: {len(df_analysis)}")

    # Map '56_monitor_postdeploy' to Binary Compliance (Yes/No)
    def map_monitoring(val):
        s = str(val).lower()
        # Strict negative filter first
        if any(x in s for x in ['no', 'none', 'not ', 'never', 'n/a', 'false', '0']):
            return 'No'
        # Positive keywords
        if any(x in s for x in ['yes', 'monitor', 'review', 'audit', 'check', 'ongoing', 'continuous', 'annual']):
            return 'Yes'
        # Default fallback (conservative)
        return 'No'

    df_analysis['Monitored'] = df_analysis['56_monitor_postdeploy'].apply(map_monitoring)

    # Generate Contingency Table
    ct = pd.crosstab(df_analysis['Phase'], df_analysis['Monitored'])
    print("\nContingency Table:")
    print(ct)

    # Calculate Rates
    rates = pd.crosstab(df_analysis['Phase'], df_analysis['Monitored'], normalize='index') * 100
    print("\nMonitoring Compliance Rates (%):")
    print(rates)

    # Statistical Test
    chi2, p, dof, ex = stats.chi2_contingency(ct)
    print(f"\nChi-Square Test Result:")
    print(f"Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # Deliverables & Insights
    try:
        op_rate = rates.loc['Operational', 'Yes']
        pre_rate = rates.loc['Pre-Operational', 'Yes']
        print(f"\nInsight: {op_rate:.1f}% of Operational systems have monitoring vs {pre_rate:.1f}% of Pre-Operational systems.")
        
        if p < 0.05:
            print("Conclusion: Significant difference detected between stages.")
        else:
            print("Conclusion: No significant difference detected.")
            
        # Check for Governance Gap
        if op_rate < 50:
            print("ALERT: Major Governance Gap. Less than 50% of operational systems are monitored.")
    except KeyError:
        print("Insufficient data to calculate specific rates.")

if __name__ == "__main__":
    run_experiment()