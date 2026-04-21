import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback to parent directory as per instructions
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Total EO13960 records: {len(eo_df)}")

# 1. Map Development Stage
operational_list = ['Operation and Maintenance', 'In production', 'In mission']
new_dev_list = ['Implementation and Assessment', 'Acquisition and/or Development', 'Initiated', 'Planned']

def map_stage(x):
    if pd.isna(x):
        return None
    val = str(x).strip()
    if val in operational_list:
        return 'Operational'
    elif val in new_dev_list:
        return 'New/Dev'
    return None  # Excludes 'Retired' and others

eo_df['stage_group'] = eo_df['16_dev_stage'].apply(map_stage)

# 2. Map AI Notice
# We define Compliance (Yes) vs Non-Compliance (No).
# We exclude cases where the requirement is N/A or Waived.

# Based on previous output, specific exclusion strings:
exclusions = [
    'N/A - individuals are not interacting with the AI for this use case',
    'AI is not safety or rights-impacting.',
    'Agency CAIO has waived this minimum practice and reported such waiver to OMB.'
]

def map_notice(x):
    if pd.isna(x):
        return None
    val = str(x).strip()
    
    # Check for exclusions
    if val in exclusions:
        return None
    
    # Check for Non-Compliance
    if 'None of the above' in val:
        return 'No'
    
    # If it's not N/A and not 'None of the above', it implies some form of notice was selected
    # (e.g. 'Online', 'Email', 'In-person', 'Other')
    return 'Yes'

eo_df['notice_compliance'] = eo_df['59_ai_notice'].apply(map_notice)

# Create analysis dataframe
analysis_df = eo_df.dropna(subset=['stage_group', 'notice_compliance']).copy()

print(f"Records available for analysis after cleaning: {len(analysis_df)}")

if len(analysis_df) < 5:
    print("Insufficient data for analysis.")
else:
    # Contingency Table
    contingency = pd.crosstab(analysis_df['stage_group'], analysis_df['notice_compliance'])
    print("\nContingency Table (Stage vs Notice Compliance):")
    print(contingency)
    
    # Rates
    rates = pd.crosstab(analysis_df['stage_group'], analysis_df['notice_compliance'], normalize='index') * 100
    print("\nCompliance Rates (%):")
    print(rates)

    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    # Visualize
    plt.figure(figsize=(8, 6))
    
    # Extract 'Yes' rates for plotting
    if 'Yes' in rates.columns:
        yes_rates = rates['Yes']
    else:
        yes_rates = pd.Series([0, 0], index=['New/Dev', 'Operational'])
        
    # Ensure both categories exist in index for plotting consistency
    for cat in ['New/Dev', 'Operational']:
        if cat not in yes_rates.index:
            yes_rates[cat] = 0
            
    # Sort for consistent order
    yes_rates = yes_rates.sort_index()
    
    bars = plt.bar(yes_rates.index, yes_rates.values, color=['skyblue', 'salmon'])
    plt.title('AI Notice Compliance by Development Stage')
    plt.ylabel('Compliance Rate (%)')
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
