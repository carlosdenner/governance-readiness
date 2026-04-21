import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load Data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    print("Error: Dataset not found.")
    exit(1)

# 2. Filter for AIID Incidents
df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# 3. Categorize Autonomy Level
# Mapping based on debug findings: Autonomy1/2 -> Low, Autonomy3 -> High
def classify_autonomy(val):
    s = str(val).lower()
    if 'autonomy3' in s:
        return 'High (Autonomous)'
    elif 'autonomy1' in s or 'autonomy2' in s:
        return 'Low (Augmented)'
    return 'Unknown'

df_aiid['Autonomy_Cat'] = df_aiid['Autonomy Level'].apply(classify_autonomy)

# 4. Categorize Harm Type
# Use 'Tangible Harm' for Physical and 'Special Interest Intangible Harm' for Intangible
def classify_harm(row):
    # Check Tangible Harm (Physical)
    tangible = str(row.get('Tangible Harm', '')).lower()
    is_physical = 'definitively' in tangible or 'imminent' in tangible
    
    # Check Intangible Harm
    intangible_col = str(row.get('Special Interest Intangible Harm', '')).lower()
    is_intangible = 'yes' in intangible_col
    
    if is_physical:
        return 'Physical'
    elif is_intangible:
        return 'Intangible'
    else:
        # Fallback: if 'no tangible harm' is explicitly stated and 'Harm Domain' is yes
        harm_domain = str(row.get('Harm Domain', '')).lower()
        if 'no tangible' in tangible and 'yes' in harm_domain:
            return 'Intangible'
            
    return 'Unknown'

df_aiid['Harm_Cat'] = df_aiid.apply(classify_harm, axis=1)

# 5. Filter for Analysis
analysis_df = df_aiid[
    (df_aiid['Autonomy_Cat'] != 'Unknown') &
    (df_aiid['Harm_Cat'] != 'Unknown')
].copy()

print(f"Data points for analysis: {len(analysis_df)}")
print("Distribution:")
print(analysis_df.groupby(['Autonomy_Cat', 'Harm_Cat']).size())

# 6. Statistical Analysis
ct = pd.crosstab(analysis_df['Autonomy_Cat'], analysis_df['Harm_Cat'])

print("\n--- Contingency Table ---")
print(ct)

if not ct.empty and ct.shape == (2, 2):
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * 1)) # min_dim - 1 is 1
    
    print("\n--- Statistical Test Results ---")
    print(f"Chi-Square: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")
    print(f"Cramer's V: {cramers_v:.4f}")
    
    alpha = 0.05
    if p < alpha:
        print("\nResult: Reject Null Hypothesis. Significant association found.")
    else:
        print("\nResult: Fail to reject Null Hypothesis. No significant association.")
        
    # 7. Visualization
    # Normalize rows to percentages
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    ax = ct_norm.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], figsize=(8, 6))
    
    plt.title('Harm Type Distribution by Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Percentage')
    plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1))
    
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
        
    plt.tight_layout()
    plt.show()

elif not ct.empty:
    print("\nNot enough categories for 2x2 Chi-Square. Printing raw counts only.")
    print(ct)
else:
    print("\nNo valid data found after filtering.")
