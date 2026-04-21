import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# Mapping functions based on debug findings
def map_autonomy(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    
    if val_str == 'Autonomy3':
        return 'High' # High Autonomy
    elif val_str in ['Autonomy1', 'Autonomy2']:
        return 'Low' # Low/Assistive Autonomy
    return np.nan

def map_harm_dist(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    
    if val_str == 'none':
        # If harm is not distributed based on a group trait, we categorize it as Individual/General
        return 'Individual'
    elif val_str == 'unclear':
        return np.nan
    else:
        # Any presence of demographic traits (race, sex, etc.) implies Collective/Group-based harm
        return 'Collective'

# Apply mappings
aiid_df['autonomy_bin'] = aiid_df['Autonomy Level'].apply(map_autonomy)
aiid_df['harm_bin'] = aiid_df['Harm Distribution Basis'].apply(map_harm_dist)

# Filter out unmapped data
analysis_df = aiid_df.dropna(subset=['autonomy_bin', 'harm_bin'])

print(f"Data points for analysis: {len(analysis_df)}")
print("Autonomy counts:")
print(analysis_df['autonomy_bin'].value_counts())
print("Harm Scale counts:")
print(analysis_df['harm_bin'].value_counts())

if len(analysis_df) == 0:
    print("No valid data points found after mapping. Exiting.")
else:
    # Create Contingency Table
    contingency_table = pd.crosstab(analysis_df['autonomy_bin'], analysis_df['harm_bin'])
    print("\nContingency Table (Count):")
    print(contingency_table)

    # Calculate Proportions (Row-wise) for plotting
    props = pd.crosstab(analysis_df['autonomy_bin'], analysis_df['harm_bin'], normalize='index')
    print("\nProportions (Row-wise):")
    print(props)

    # Perform Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    # Interpret results
    alpha = 0.05
    if p < alpha:
        print("\nConclusion: Reject Null Hypothesis. There is a significant association between Autonomy Level and Harm Scale.")
    else:
        print("\nConclusion: Fail to Reject Null Hypothesis. No significant association found.")

    # Visualization
    # Colors: Collective (Red/Orange), Individual (Blue/Green)
    # Check column order for colors. Usually alphabetical: Collective, Individual.
    colors = ['#d62728', '#1f77b4'] 
    
    ax = props.plot(kind='bar', stacked=True, color=colors, figsize=(8, 6))
    plt.title('Harm Distribution Scale by AI Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Proportion of Incidents')
    plt.legend(title='Harm Scale', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Add labels
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', label_type='center', color='white')

    plt.show()