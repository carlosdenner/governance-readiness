import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Total EO 13960 records: {len(eo_df)}")

# --- Step 1: Analyze and Map Lifecycle Stage ---
# Column: 16_dev_stage
stage_col = '16_dev_stage'
print(f"\nUnique values in '{stage_col}':")
print(eo_df[stage_col].unique())

def map_stage(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    # Operational keywords
    if any(x in val_lower for x in ['operation', 'maintenance', 'use', 'implemented', 'deployed']):
        return 'Operational'
    # Development keywords
    if any(x in val_lower for x in ['development', 'acquisition', 'planning', 'research', 'pilot', 'test']):
        return 'Development'
    return None

eo_df['lifecycle_category'] = eo_df[stage_col].apply(map_stage)

# --- Step 2: Analyze and Map Impact Assessment ---
# Column: 52_impact_assessment
assess_col = '52_impact_assessment'
print(f"\nUnique values in '{assess_col}':")
print(eo_df[assess_col].unique())

def map_assessment(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    if 'yes' in val_lower:
        return 'Yes'
    if 'no' in val_lower or 'not' in val_lower:
        return 'No'
    return None

eo_df['has_impact_assessment'] = eo_df[assess_col].apply(map_assessment)

# --- Step 3: Filter and Create Contingency Table ---
analysis_df = eo_df.dropna(subset=['lifecycle_category', 'has_impact_assessment'])

print(f"\nRecords after cleaning and mapping: {len(analysis_df)}")

contingency_table = pd.crosstab(
    analysis_df['lifecycle_category'],
    analysis_df['has_impact_assessment']
)

print("\nContingency Table (Count):")
print(contingency_table)

# Calculate rates
rates = pd.crosstab(
    analysis_df['lifecycle_category'],
    analysis_df['has_impact_assessment'],
    normalize='index'
) * 100

print("\nCompliance Rates (%):")
print(rates)

# --- Step 4: Statistical Test ---
if contingency_table.size > 0:
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Degrees of Freedom: {dof}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Statistically Significant (Reject Null Hypothesis)")
        print("Interpretation: There is a significant association between lifecycle stage and impact assessment compliance.")
    else:
        print("Result: Not Statistically Significant (Fail to Reject Null)")
        print("Interpretation: No significant difference in impact assessment compliance found between stages.")
        
    # Visualization
    try:
        rates.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'])
        plt.title('Impact Assessment Compliance by Lifecycle Stage')
        plt.ylabel('Percentage')
        plt.xlabel('Lifecycle Stage')
        plt.xticks(rotation=0)
        plt.legend(title='Impact Assessment', loc='upper right')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")
else:
    print("Insufficient data for statistical testing.")