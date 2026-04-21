import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Clean and Group Lifecycle Stage (16_dev_stage)
def map_stage(stage):
    if pd.isna(stage):
        return None
    stage_lower = str(stage).lower()
    if any(x in stage_lower for x in ['use', 'operation', 'maintenance', 'retired']):
        return 'Operation/Use'
    elif any(x in stage_lower for x in ['plan', 'develop', 'pilot', 'acquisition']):
        return 'Pilot/Planned'
    else:
        return 'Other/Unknown'

eo_data['Stage_Group'] = eo_data['16_dev_stage'].apply(map_stage)

# Filter out Unknown/None stages for the analysis
analysis_df = eo_data[eo_data['Stage_Group'].isin(['Operation/Use', 'Pilot/Planned'])].copy()

# Clean and Binarize 'Timely Resources' (47_timely_resources)
def map_resources(val):
    if pd.isna(val):
        return 0
    val_lower = str(val).lower()
    if 'yes' in val_lower:
        return 1
    return 0

analysis_df['Has_Resources'] = analysis_df['47_timely_resources'].apply(map_resources)

# Generate Contingency Table
contingency_table = pd.crosstab(analysis_df['Stage_Group'], analysis_df['Has_Resources'])
contingency_table.columns = ['No/Unknown', 'Yes (Confirmed)']
print("\nContingency Table (Stage vs Timely Resources):")
print(contingency_table)

# Calculate Rates
rates = analysis_df.groupby('Stage_Group')['Has_Resources'].mean()
counts = analysis_df.groupby('Stage_Group')['Has_Resources'].count()
success_counts = analysis_df.groupby('Stage_Group')['Has_Resources'].sum()

print("\nResource Confirmation Rates:")
print(rates)

# Perform Chi-square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square Test Results:")
print(f"Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Visualization
plt.figure(figsize=(8, 6))
bar_colors = ['#ff9999', '#66b3ff']
ax = rates.plot(kind='bar', color=bar_colors, alpha=0.8, edgecolor='black')

plt.title("Proportion of AI Systems with Confirmed 'Timely Resources' by Stage")
plt.ylabel('Proportion (Yes / Total)')
plt.xlabel('Lifecycle Stage Group')
plt.ylim(0, 1.1)

# Add value labels using .iloc for positional indexing
for i, v in enumerate(rates):
    # Fix: use .iloc to access values by integer position since the Series index is string-based
    n_success = success_counts.iloc[i]
    n_total = counts.iloc[i]
    ax.text(i, v + 0.02, f"{v:.1%} (n={n_success}/{n_total})", 
            ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
