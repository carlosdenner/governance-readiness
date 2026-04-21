import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback for local testing if needed, though instruction says one level above
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for 'eo13960_scored' source table
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Identify relevant columns
# Based on metadata, likely names are '52_impact_assessment' and '55_independent_eval'
col_impact = '52_impact_assessment'
col_eval = '55_independent_eval'

# Check if columns exist
if col_impact not in eo_data.columns or col_eval not in eo_data.columns:
    # search for columns if exact names don't match
    cols = eo_data.columns.tolist()
    col_impact = next((c for c in cols if 'impact_assessment' in c.lower()), None)
    col_eval = next((c for c in cols if 'independent_eval' in c.lower()), None)

print(f"Using columns: '{col_impact}' and '{col_eval}'")

# Inspect unique values to determine mapping
print("Unique values in Impact Assessment:", eo_data[col_impact].unique())
print("Unique values in Independent Eval:", eo_data[col_eval].unique())

# Function to map values to binary
def map_to_binary(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip().lower()
    if val_str.startswith('yes'):
        return 1
    return 0

# Apply mapping
eo_data['has_impact_assessment'] = eo_data[col_impact].apply(map_to_binary)
eo_data['has_independent_eval'] = eo_data[col_eval].apply(map_to_binary)

# Create Contingency Table
contingency_table = pd.crosstab(
    eo_data['has_impact_assessment'], 
    eo_data['has_independent_eval'], 
    rownames=['Impact Assessment'], 
    colnames=['Independent Eval']
)

print("\nContingency Table (0=No/NA, 1=Yes):")
print(contingency_table)

# Perform Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Calculate Phi Coefficient
n = contingency_table.sum().sum()
phi = np.sqrt(chi2 / n)

print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Phi Coefficient: {phi:.4f}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nResult: Statistically significant association found (Reject Null Hypothesis).")
    if phi > 0.5:
        print("Strength: Strong association.")
    elif phi > 0.3:
        print("Strength: Moderate association.")
    else:
        print("Strength: Weak association.")
else:
    print("\nResult: No statistically significant association found (Fail to reject Null Hypothesis).")
    print("This supports the 'Governance-Silo' hypothesis that controls are applied piecemeal.")

# Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Co-occurrence of Governance Controls')
plt.ylabel('Has Impact Assessment')
plt.xlabel('Has Independent Evaluation')
plt.tight_layout()
plt.show()
