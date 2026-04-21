import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
# Using current directory based on previous context, ignoring the relative path instruction if it failed previously
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Fallback if the file is indeed in the parent directory as hinted in the prompt
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {eo_df.shape}")

# Helper function to determine status
def get_status(row):
    stage = str(row.get('16_dev_stage', '')).lower()
    date_ret = row.get('21_date_retired', np.nan)
    
    # Check explicit stage labels
    if 'retired' in stage or 'decommissioned' in stage:
        return 'Retired'
    
    # Check if a retirement date exists
    # If it's a string not equal to 'nan', or a valid number
    if pd.notna(date_ret) and str(date_ret).lower() != 'nan' and str(date_ret).strip() != '':
        return 'Retired'
    
    return 'Active'

# Helper function to determine impact type
def get_impact(row):
    impact = str(row.get('17_impact_type', '')).lower()
    if 'rights' in impact:
        return 'Rights-Impacting'
    return 'Other'

# Apply classifications
eo_df['status'] = eo_df.apply(get_status, axis=1)
eo_df['impact_category'] = eo_df.apply(get_impact, axis=1)

# Generate Contingency Table
contingency = pd.crosstab(eo_df['status'], eo_df['impact_category'])
print("\nContingency Table (Count):")
print(contingency)

# Calculate percentages
rates = pd.crosstab(eo_df['status'], eo_df['impact_category'], normalize='index') * 100
print("\nDistribution Rates (%):")
print(rates)

# Statistical Tests
if contingency.shape == (2, 2):
    # Chi-Square Test
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test: Statistic={chi2:.4f}, p-value={p:.6e}")
    
    # Fisher's Exact Test for Odds Ratio
    # We want to check if Retired are more likely to be Rights-Impacting
    # Table structure usually: [[A, B], [C, D]]
    # Let's align it explicitly:
    #              Rights-Impacting   Other
    # Retired      a                  b
    # Active       c                  d
    
    try:
        a = contingency.loc['Retired', 'Rights-Impacting']
        b = contingency.loc['Retired', 'Other']
        c = contingency.loc['Active', 'Rights-Impacting']
        d = contingency.loc['Active', 'Other']
        
        table_ordered = [[a, b], [c, d]]
        odds_ratio, p_fisher = stats.fisher_exact(table_ordered)
        print(f"Fisher's Exact Test: Odds Ratio={odds_ratio:.4f}, p-value={p_fisher:.6e}")
        print(f"Interpretation: Retired systems are {odds_ratio:.2f}x as likely to be Rights-Impacting as Active systems (in terms of odds).")
    except KeyError:
        print("KeyError: Could not construct 2x2 table properly (missing categories).")
else:
    print("Contingency table is not 2x2. Skipping statistical tests.")

# Visualization
plt.figure(figsize=(10, 6))
# Plotting
ax = rates.plot(kind='bar', stacked=True, color=['#d62728', '#1f77b4'], alpha=0.8)
plt.title('Proportion of Rights-Impacting Systems: Active vs. Retired')
plt.xlabel('System Status')
plt.ylabel('Percentage')
plt.legend(title='Impact Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()