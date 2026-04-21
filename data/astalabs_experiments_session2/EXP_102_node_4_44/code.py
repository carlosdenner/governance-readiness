import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Step 1: Clean '10_commercial_ai' (System Source) ---
# Logic: 'None of the above.' implies the system is not one of the standard COTS tools listed, 
# hence categorized as 'Custom/Federal' for this comparison. 
# Specific string values imply 'Commercial' use cases.

def classify_source(val):
    if pd.isna(val):
        return np.nan
    s_val = str(val).strip()
    if s_val == 'None of the above.':
        return 'Custom/Federal'
    else:
        return 'Commercial'

df_eo['system_source'] = df_eo['10_commercial_ai'].apply(classify_source)

# Drop rows where source is unknown (NaN in 10_commercial_ai)
df_analysis = df_eo.dropna(subset=['system_source']).copy()

# --- Step 2: Clean '56_monitor_postdeploy' (Monitoring Status) ---
# Logic: Identify positive assertions of monitoring infrastructure.
# NaNs are treated as 'No/Not Reported' in this context of survey compliance.

def classify_monitoring(val):
    if pd.isna(val):
        return 'No'
    
    s_val = str(val).lower()
    
    # Positive keywords based on unique values analysis
    # "Intermittent and Manually Updated..."
    # "Automated and Regularly Scheduled..."
    # "Established Process..."
    if any(keyword in s_val for keyword in ['intermittent', 'automated', 'established', 'manually updated']):
        return 'Yes'
    
    # Negative keywords: "No monitoring protocols", "not safety impacting"
    # "under development" is treated as No (not currently active)
    return 'No'

df_analysis['has_monitoring'] = df_analysis['56_monitor_postdeploy'].apply(classify_monitoring)

# --- Step 3: Analysis ---
print(f"Records for analysis: {len(df_analysis)}")
print("System Source Distribution:\n", df_analysis['system_source'].value_counts())
print("Monitoring Status Distribution:\n", df_analysis['has_monitoring'].value_counts())

# Contingency Table
contingency_table = pd.crosstab(df_analysis['system_source'], df_analysis['has_monitoring'])
print("\nContingency Table (Source vs Monitoring):")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Calculate Odds Ratio
# OR = (a*d) / (b*c)
# Table structure usually: 
#              No   Yes
# Commercial   a    b
# Custom       c    d
# But let's check the columns of crosstab first

if 'Yes' in contingency_table.columns and 'No' in contingency_table.columns:
    # Calculate percentage of 'Yes' for each group
    monitoring_rates = pd.crosstab(df_analysis['system_source'], df_analysis['has_monitoring'], normalize='index') * 100
    print("\nMonitoring Rates (%):")
    print(monitoring_rates)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Ensure order is Commercial then Custom for comparison
    sources = ['Commercial', 'Custom/Federal']
    # Handle case if one source is missing
    present_sources = [s for s in sources if s in monitoring_rates.index]
    
    yes_rates = monitoring_rates.loc[present_sources, 'Yes']
    
    bars = plt.bar(present_sources, yes_rates, color=['#ff7f0e', '#1f77b4'])
    
    plt.title('Post-Deployment Monitoring Rates: Commercial vs Custom AI')
    plt.xlabel('System Source')
    plt.ylabel('Reported Monitoring (%)')
    plt.ylim(0, max(yes_rates.max() * 1.2, 10))
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data to generate 'Yes' column for monitoring.")
