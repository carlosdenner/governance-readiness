import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)}")

# 3. Define Mapping Functions

def map_autonomy(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    if 'autonomy3' in val_str:
        return 'High Autonomy'
    elif 'autonomy1' in val_str or 'autonomy2' in val_str:
        return 'Low/Medium Autonomy'
    return np.nan

def map_harm_composite(row):
    # Extract values
    tangible = str(row['Tangible Harm']).lower()
    intangible = str(row['Special Interest Intangible Harm']).lower()
    
    # Check for Physical/Tangible Harm indicators
    # We include definitive occurrences and imminent risks (near misses) as 'Physical/Tangible'
    is_tangible = False
    if 'tangible harm definitively occurred' in tangible:
        is_tangible = True
    elif 'imminent risk' in tangible:
        is_tangible = True
        
    # Check for Intangible Harm
    is_intangible = False
    if 'yes' in intangible:
        is_intangible = True
        
    # Classification Logic
    if is_tangible:
        return 'Physical/Tangible Harm'
    elif is_intangible:
        return 'Intangible Harm'
    else:
        return np.nan

# 4. Apply Mappings
aiid_df['Autonomy_Clean'] = aiid_df['Autonomy Level'].apply(map_autonomy)
aiid_df['Harm_Clean'] = aiid_df.apply(map_harm_composite, axis=1)

# 5. Filter for Analysis
analysis_df = aiid_df.dropna(subset=['Autonomy_Clean', 'Harm_Clean'])

print(f"\nRecords ready for analysis: {len(analysis_df)}")
print("Distribution by Autonomy:\n", analysis_df['Autonomy_Clean'].value_counts())
print("Distribution by Harm:\n", analysis_df['Harm_Clean'].value_counts())

if len(analysis_df) < 5:
    print("Insufficient data for statistical testing.")
else:
    # 6. Statistical Analysis
    contingency_table = pd.crosstab(analysis_df['Autonomy_Clean'], analysis_df['Harm_Clean'])
    print("\nContingency Table:")
    print(contingency_table)

    # Chi-square
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")

    # Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    print(f"Cramer's V: {cramers_v:.4f}")

    # 7. Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Oranges')
    plt.title('Autonomy Level vs. Harm Type')
    plt.ylabel('Autonomy Level')
    plt.xlabel('Harm Type')
    plt.tight_layout()
    plt.show()

    # Row Percentages
    row_props = pd.crosstab(analysis_df['Autonomy_Clean'], analysis_df['Harm_Clean'], normalize='index') * 100
    print("\nRow Proportions (%):")
    print(row_props)
