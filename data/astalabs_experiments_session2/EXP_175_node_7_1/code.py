import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
print(f"Loading dataset from {file_path}...")

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {eo_df.shape}")

col_assessment = '52_impact_assessment'
col_mitigation = '62_disparity_mitigation'

# Function to clean and binary encode text responses
def clean_binary_response(text):
    if pd.isna(text):
        return np.nan
    t = str(text).lower().strip()
    # Positive indicators
    if t.startswith('yes') or 'completed' in t or 'conducted' in t or 'planned' in t:
        return 'Yes'
    # Negative indicators
    if t.startswith('no') or 'none' in t or 'n/a' in t or 'not applicable' in t or 'not required' in t:
        return 'No'
    return np.nan

# Apply cleaning
eo_df['assessment_clean'] = eo_df[col_assessment].apply(clean_binary_response)
eo_df['mitigation_clean'] = eo_df[col_mitigation].apply(clean_binary_response)

# Drop rows where either value could not be determined
valid_df = eo_df.dropna(subset=['assessment_clean', 'mitigation_clean'])
print(f"Valid data points after cleaning: {len(valid_df)}")

# Generate Contingency Table
contingency_table = pd.crosstab(valid_df['assessment_clean'], valid_df['mitigation_clean'])

# Ensure 2x2 table by reindexing (handling missing categories like 'Yes' in mitigation)
contingency_table = contingency_table.reindex(index=['No', 'Yes'], columns=['No', 'Yes'], fill_value=0)

print("\nContingency Table (Impact Assessment vs Disparity Mitigation):")
print(contingency_table)

# Calculate counts and rates safely
assessed_yes = contingency_table.loc['Yes'].sum()
mitigated_given_assessed = contingency_table.loc['Yes', 'Yes']

assessed_no = contingency_table.loc['No'].sum()
mitigated_given_not_assessed = contingency_table.loc['No', 'Yes']

rate_assessed = (mitigated_given_assessed / assessed_yes * 100) if assessed_yes > 0 else 0.0
rate_not_assessed = (mitigated_given_not_assessed / assessed_no * 100) if assessed_no > 0 else 0.0

print(f"\nMitigation Rate when Assessment='Yes': {rate_assessed:.2f}% ({mitigated_given_assessed}/{assessed_yes})")
print(f"Mitigation Rate when Assessment='No': {rate_not_assessed:.2f}% ({mitigated_given_not_assessed}/{assessed_no})")

# Analysis of the 'Assessment-Action Gap'
# Gap defined as Assessment='Yes' but Mitigation='No'
gap_count = contingency_table.loc['Yes', 'No']
gap_percentage = (gap_count / assessed_yes * 100) if assessed_yes > 0 else 0.0

print(f"\nAssessment-Action Gap Analysis:")
print(f"Number of systems with Impact Assessment but NO Disparity Mitigation: {gap_count}")
print(f"Percentage of Assessed systems that are Unmitigated ('Paper Tigers'): {gap_percentage:.2f}%")

# Chi-Square Test
# Check if we have enough data to run the test (at least 2 dimensions with some data)
if contingency_table.sum().sum() > 0:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")
    
    # Interpretation
    alpha = 0.05
    if p < alpha:
        print("\nResult: Statistically significant relationship found.")
        if rate_assessed > rate_not_assessed:
            print("Evidence supports: Conducting an Impact Assessment positively correlates with Disparity Mitigation.")
        else:
            print("Evidence suggests: Negative or paradoxical relationship.")
    else:
        print("\nResult: No statistically significant relationship found.")
        print("Interpretation: Conducting an Impact Assessment does not statistically guarantee Disparity Mitigation plans (supports the 'Assessment-Action Gap').")
else:
    print("\nInsufficient data for Chi-Square test.")

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Impact Assessment vs. Disparity Mitigation')
plt.xlabel('Disparity Mitigation Planned?')
plt.ylabel('Impact Assessment Conducted?')
plt.show()
