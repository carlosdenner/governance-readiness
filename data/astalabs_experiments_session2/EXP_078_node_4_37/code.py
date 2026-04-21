import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Cleaning Functions ---

def clean_ato(val):
    s = str(val).strip().lower()
    if s == 'yes':
        return 'Has ATO'
    if 'approved enclave' in s:
        return 'Has ATO'
    return 'No ATO'

def clean_monitoring(val):
    s = str(val).strip().lower()
    # Positive indicators
    if 'automated' in s or 'established' in s or 'intermittent' in s:
        return 'Monitored'
    # Negative indicators (explicit 'no', 'under development', or irrelevant justification)
    # 'nan' will fall through to here as it doesn't contain positive keywords
    return 'Not Monitored'

# Apply cleaning
eo_data['ato_status'] = eo_data['40_has_ato'].apply(clean_ato)
eo_data['monitoring_status'] = eo_data['56_monitor_postdeploy'].apply(clean_monitoring)

# Debug: Check distribution
print("ATO Status Distribution:")
print(eo_data['ato_status'].value_counts())
print("\nMonitoring Status Distribution:")
print(eo_data['monitoring_status'].value_counts())

# Create Contingency Table
contingency_table = pd.crosstab(eo_data['ato_status'], eo_data['monitoring_status'])
print("\nContingency Table:")
print(contingency_table)

# Calculate percentages for visualization
# We want P(Monitored | ATO Status)
ato_summary = pd.crosstab(eo_data['ato_status'], eo_data['monitoring_status'], normalize='index') * 100
print("\nMonitoring Percentages by ATO Status:")
print(ato_summary)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Visualization
if 'Monitored' in ato_summary.columns:
    plt.figure(figsize=(10, 6))
    # Plotting the 'Monitored' percentage
    # Reorder index if needed to have Has ATO vs No ATO
    desired_order = ['Has ATO', 'No ATO']
    # Filter to only existing indices
    existing_order = [x for x in desired_order if x in ato_summary.index]
    
    plot_data = ato_summary.loc[existing_order, 'Monitored']
    
    bars = plt.bar(plot_data.index, plot_data.values, color=['#4CAF50', '#F44336'])
    
    plt.title('Impact of ATO Authorization on Post-Deployment Monitoring')
    plt.xlabel('Authorization Status')
    plt.ylabel('Percentage of Systems with Monitoring (%)')
    plt.ylim(0, 100)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("No 'Monitored' category found to plot.")
