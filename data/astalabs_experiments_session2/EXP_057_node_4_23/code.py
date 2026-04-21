import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback for different execution environments if needed, though instruction says use given dataset
    df = pd.read_csv(f'../{file_path}', low_memory=False)

# Filter for EO 13960 subset
subset = df[df['source_table'] == 'eo13960_scored'].copy()

# Clean and Map Impact Type
def map_impact(val):
    val_str = str(val).strip()
    if val_str in ['Both', 'Rights-Impacting', 'Safety-Impacting', 'Safety-impacting', 'Rights-Impacting\n']:
        return 'High Impact'
    elif val_str in ['Neither', 'Low', 'None']:
        return 'Low Impact'
    return 'Unknown'

# Clean and Map Assessment Status
def map_assessment(val):
    val_str = str(val).strip().upper()
    if 'YES' in val_str:
        return 'Yes'
    elif 'NO' in val_str or 'PLANNED' in val_str:
        return 'No'
    return 'Unknown'

subset['impact_tier'] = subset['17_impact_type'].apply(map_impact)
subset['assessment_done'] = subset['52_impact_assessment'].apply(map_assessment)

# Filter for valid analysis rows
analysis_df = subset[
    (subset['impact_tier'] != 'Unknown') & 
    (subset['assessment_done'] != 'Unknown')
].copy()

print(f"Analysis Subset Shape: {analysis_df.shape}")
print("\nDistribution of Impact Tiers:")
print(analysis_df['impact_tier'].value_counts())

# Create Contingency Table
contingency_table = pd.crosstab(analysis_df['impact_tier'], analysis_df['assessment_done'])
print("\nContingency Table (Impact Tier vs Assessment Completion):")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:\nStatistic: {chi2:.4f}\np-value: {p:.4e}\nDoF: {dof}")

# Calculate Compliance Rates
rates = analysis_df.groupby('impact_tier')['assessment_done'].value_counts(normalize=True).unstack().fillna(0)
compliance_rates = rates['Yes'] * 100
print("\nCompliance Rates (% Assessment Completed):")
print(compliance_rates)

# Visualization
plt.figure(figsize=(10, 6))
# Calculate percentages for the plot
prop_df = (analysis_df.groupby(['impact_tier'])['assessment_done']
           .value_counts(normalize=True)
           .rename('percentage')
           .reset_index())
prop_df['percentage'] *= 100

sns.barplot(x='impact_tier', y='percentage', hue='assessment_done', data=prop_df, palette='viridis')
plt.title('Impact Assessment Compliance by Risk Tier')
plt.ylabel('Percentage of Use Cases (%)')
plt.xlabel('Impact Tier')
plt.ylim(0, 100)
plt.legend(title='Assessment Completed')

# Annotate bars
for p in plt.gca().patches:
    txt = f"{p.get_height():.1f}%"
    plt.gca().text(p.get_x() + p.get_width()/2, p.get_height() + 1, txt, ha='center')

plt.tight_layout()
plt.show()
