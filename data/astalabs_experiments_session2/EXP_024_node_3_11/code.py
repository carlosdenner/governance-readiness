import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the dataset
file_path = "../astalabs_discovery_all_data.csv"
try:
    # Using low_memory=False to avoid DtypeWarning, or specifying types if known. 
    # Given the sparse nature, we just load and then filter.
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if file is in current directory (just in case)
    df = pd.read_csv("astalabs_discovery_all_data.csv", low_memory=False)

# Filter for source_table == 'eo13960_scored'
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

print("Initial EO 13960 records:", len(df_eo))

# Inspect unique values for mapping
print("\nUnique values in '16_dev_stage':")
print(df_eo['16_dev_stage'].unique())

print("\nUnique values in '52_impact_assessment':")
print(df_eo['52_impact_assessment'].unique())

# Clean and Map Development Stage
# Hypothesis: 'Operation and Maintenance' vs 'Development/Acquisition'
# Let's define the buckets based on typical values found in EO 13960 datasets
# Common values: 'Operation and maintenance', 'Development and acquisition', 'Planned'

def map_stage(stage):
    if pd.isna(stage):
        return np.nan
    stage = str(stage).lower()
    if 'operation' in stage or 'maintenance' in stage or 'deployed' in stage:
        return 'Operation'
    elif 'develop' in stage or 'acquisition' in stage or 'plan' in stage or 'pilot' in stage:
        return 'Development'
    else:
        return 'Other/Unknown'

df_eo['stage_category'] = df_eo['16_dev_stage'].apply(map_stage)

# Clean and Map Impact Assessment
# Usually 'Yes', 'No', or specific description. We treat non-null/non-no as Yes? 
# Or check for affirmative keywords.
# Let's assume strict 'Yes' vs others first, but will refine based on print output above if needed.
# For now, a generic mapper.

def map_impact(val):
    if pd.isna(val):
        return 'No'
    val_str = str(val).lower().strip()
    if val_str in ['yes', 'true', '1']:
        return 'Yes'
    # Sometimes it might contain a link or text. If it looks like a boolean field, we stick to Yes/No.
    # If the column contains text descriptions, we might need a more heuristic approach.
    # Based on metadata '52_impact_assessment' often implies a boolean or link.
    # We'll check if it starts with 'yes' or has content implying existence.
    if val_str.startswith('yes'):
        return 'Yes'
    return 'No'

df_eo['has_impact_assessment'] = df_eo['52_impact_assessment'].apply(map_impact)

# Filter out Unknown stages
df_analysis = df_eo[df_eo['stage_category'].isin(['Operation', 'Development'])].copy()

print("\nRecords after stage filtering:", len(df_analysis))
print(df_analysis['stage_category'].value_counts())
print(df_analysis['has_impact_assessment'].value_counts())

# Create Contingency Table
contingency_table = pd.crosstab(df_analysis['stage_category'], df_analysis['has_impact_assessment'])
print("\nContingency Table:")
print(contingency_table)

# Check if we have enough data
if contingency_table.size == 4:
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Calculate percentages
    # Row-wise normalization to see % compliant per stage
    pct_table = pd.crosstab(df_analysis['stage_category'], df_analysis['has_impact_assessment'], normalize='index') * 100
    print("\nPercentage Table (Row-wise):")
    print(pct_table)

    # Plot
    try:
        compliance_rates = pct_table['Yes']
    except KeyError:
        compliance_rates = pd.Series([0, 0], index=['Development', 'Operation'])

    plt.figure(figsize=(8, 6))
    bars = plt.bar(compliance_rates.index, compliance_rates.values, color=['skyblue', 'salmon'])
    plt.ylabel('Percentage with Impact Assessment (%)')
    plt.title('Impact Assessment Compliance by Development Stage')
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data for Chi-square test.")
