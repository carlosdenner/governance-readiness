import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Define columns
col_dev = '22_dev_method'
col_mitig = '62_disparity_mitigation'

# 1. Map Development Method (Independent Variable)
def map_dev_method(val):
    s = str(val).lower().strip()
    if 'contracting' in s and 'both' not in s:
        return 'Commercial/Vendor'
    elif 'in-house' in s and 'both' not in s:
        return 'Internal/Gov'
    return 'Other/Mixed'

df_eo['dev_source'] = df_eo[col_dev].apply(map_dev_method)

# Filter for only the two distinct groups
df_analysis = df_eo[df_eo['dev_source'].isin(['Commercial/Vendor', 'Internal/Gov'])].copy()

# 2. Map Disparity Mitigation (Dependent Variable)
def map_mitigation(val):
    if pd.isna(val):
        return 'No/Unknown'
    
    s = str(val).lower().strip()
    
    # List of phrases indicating NO mitigation or Not Applicable
    negative_indicators = [
        'nan',
        'n/a',
        'not applicable',
        'none',
        'no demographic',
        'not safety',
        'not rights',
        'tbd',
        'unknown'
    ]
    
    # Check if the text starts with negative phrases or matches exactly
    if any(s.startswith(x) for x in negative_indicators):
        return 'No/Unknown'
    
    # If the text is just "no" or very short negative
    if len(s) < 5 and 'no' in s:
        return 'No/Unknown'
        
    # Otherwise, assume the presence of descriptive text implies evidence
    return 'Documented'

df_analysis['mitigation_status'] = df_analysis[col_mitig].apply(map_mitigation)

# 3. Generate Statistics
print(f"\n--- Analysis Dataset (n={len(df_analysis)}) ---")
contingency_table = pd.crosstab(df_analysis['dev_source'], df_analysis['mitigation_status'])
print("\n--- Contingency Table ---")
print(contingency_table)

# Calculate percentages
props = pd.crosstab(df_analysis['dev_source'], df_analysis['mitigation_status'], normalize='index')
print("\n--- Proportions ---")
print(props)

# 4. Statistical Test (Chi-Square)
if contingency_table.size > 0:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}\nP-value: {p:.4e}\nDegrees of Freedom: {dof}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Significant difference found. Hypothesis supported or rejected based on direction.")
    else:
        print("Result: No significant difference found.")
else:
    print("Insufficient data for statistical test.")

# 5. Visualization
plt.figure(figsize=(10, 6))
ax = props.plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728'], figsize=(8, 6))

plt.title('Disparity Mitigation Documentation by Dev Source')
plt.xlabel('Development Source')
plt.ylabel('Proportion of Systems')
plt.legend(title='Mitigation Evidence', loc='upper right', bbox_to_anchor=(1.25, 1))
plt.xticks(rotation=0)

# Annotate bars
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', weight='bold')

plt.tight_layout()
plt.show()
