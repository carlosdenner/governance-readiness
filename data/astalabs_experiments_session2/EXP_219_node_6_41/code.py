import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

# Define the filename
filename = 'astalabs_discovery_all_data.csv'

# Check if file exists in current directory, if not try parent
if os.path.exists(filename):
    file_path = filename
elif os.path.exists(f'../{filename}'):
    file_path = f'../{filename}'
else:
    print(f"Error: {filename} not found in current or parent directory.")
    # List current dir for debugging purposes in case of failure
    print(f"Current dir content: {os.listdir('.')}")
    sys.exit(1)

print(f"Loading dataset from: {file_path}")

# Load the dataset
try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Filter for EO13960 data
sub_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 records loaded: {len(sub_df)}")

# ---------------------------------------------------------
# Data Preparation: Lifecycle Stage
# ---------------------------------------------------------
# Inspect unique values to ensure correct mapping
print("\nDistribution of '16_dev_stage':")
stage_counts = sub_df['16_dev_stage'].value_counts(dropna=False)
print(stage_counts)

def map_stage(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    # Operation & Maintenance stages
    if 'operation' in val_str or 'maintenance' in val_str or 'use' in val_str:
        return 'Operations (Legacy)'
    # Development & Implementation stages
    elif 'development' in val_str or 'implementation' in val_str or 'acquisition' in val_str or 'planning' in val_str:
        return 'Dev/Implementation (New)'
    else:
        return None # Exclude 'Retired' or other unclear stages

sub_df['Lifecycle_Group'] = sub_df['16_dev_stage'].apply(map_stage)

# Drop rows where Lifecycle Group is undefined
sub_df = sub_df.dropna(subset=['Lifecycle_Group'])
print(f"\nRecords after filtering for relevant lifecycle stages: {len(sub_df)}")
print(sub_df['Lifecycle_Group'].value_counts())

# ---------------------------------------------------------
# Data Preparation: Impact Assessment
# ---------------------------------------------------------
# Map to Binary: Yes vs Not Yes (No, N/A, NaN)
def map_assessment(val):
    if pd.isna(val):
        return 'No'
    val_str = str(val).strip().lower()
    if val_str == 'yes':
        return 'Yes'
    return 'No'

sub_df['Has_Assessment'] = sub_df['52_impact_assessment'].apply(map_assessment)
print("\nImpact Assessment Distribution:")
print(sub_df['Has_Assessment'].value_counts())

# ---------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------
# Create Contingency Table
contingency_table = pd.crosstab(sub_df['Lifecycle_Group'], sub_df['Has_Assessment'])
print("\nContingency Table (Count):")
print(contingency_table)

# Calculate Percentages
contingency_pct = pd.crosstab(sub_df['Lifecycle_Group'], sub_df['Has_Assessment'], normalize='index') * 100
print("\nContingency Table (Percentage):")
print(contingency_pct)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nResult: Statistically Significant Association found (Reject H0).")
else:
    print("\nResult: No Statistically Significant Association found (Fail to reject H0).")

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
# Plotting the percentage of 'Yes' for Impact Assessment by Group
yes_rates = contingency_pct['Yes'] if 'Yes' in contingency_pct.columns else pd.Series([0,0], index=contingency_pct.index)

plt.figure(figsize=(10, 6))
colors = ['#d62728', '#1f77b4'] # Red for Dev, Blue for Ops (or vice versa depending on sort)
ax = yes_rates.plot(kind='bar', color=colors, alpha=0.8)

plt.title('Impact Assessment Compliance by Lifecycle Stage')
plt.ylabel('Percentage with Completed Impact Assessment (%)')
plt.xlabel('Lifecycle Stage')
plt.ylim(0, max(yes_rates.max() * 1.2, 10)) # Add some headroom
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for i, v in enumerate(yes_rates):
    ax.text(i, v + 0.2, f"{v:.1f}%", ha='center', fontweight='bold')

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
