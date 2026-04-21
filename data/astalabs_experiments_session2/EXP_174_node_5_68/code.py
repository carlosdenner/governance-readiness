import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# Load dataset
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: Could not find dataset at {file_path}")
    exit(1)

# Filter for EO 13960 Scored subset
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# 1. Map Procurement Type (Commercial vs Custom)
# Logic: 'None of the above.' implies the use case is not a standard commercial tool -> Custom/Mission Specific.
# Specific descriptions (e.g., 'Scheduling meetings') -> Commercial.
def map_procurement(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == 'None of the above.':
        return 'Custom'
    else:
        return 'Commercial'

df_eo['Procurement_Type'] = df_eo['10_commercial_ai'].apply(map_procurement)

# Drop rows where procurement type is unknown
df_analysis = df_eo.dropna(subset=['Procurement_Type']).copy()

# 2. Map Documentation Transparency (Has_Documentation)
# Logic: Identify positive confirmation of documentation while excluding explicit negatives.
def map_documentation(x):
    if pd.isna(x):
        return 0
    s = str(x).lower().strip()
    
    # Explicit negatives
    if 'missing' in s or 'not available' in s or 'not reported' in s or s == 'no':
        return 0
    
    # Positives
    # "partial" is considered compliant/present for the purpose of "Availability of documentation"
    if ('complete' in s or 
        'partial' in s or 
        'widely' in s or 
        'yes' in s or 
        'documented' in s or 
        'is available' in s):
        return 1
        
    return 0

df_analysis['Has_Documentation'] = df_analysis['34_data_docs'].apply(map_documentation)

# 3. Generate Summary Stats
print(f"Analyzed rows: {len(df_analysis)}")
print("Distribution of Procurement Type:")
print(df_analysis['Procurement_Type'].value_counts())

print("\nDistribution of Documentation Status:")
print(df_analysis['Has_Documentation'].value_counts())

# 4. Contingency Table
contingency = pd.crosstab(df_analysis['Procurement_Type'], df_analysis['Has_Documentation'])
# Ensure columns are 0 and 1
if 0 not in contingency.columns: contingency[0] = 0
if 1 not in contingency.columns: contingency[1] = 0
contingency = contingency[[0, 1]]
contingency.columns = ['No Doc', 'Has Doc']

print("\nContingency Table:")
print(contingency)

# 5. Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# 6. Documentation Rates
rates = df_analysis.groupby('Procurement_Type')['Has_Documentation'].mean()
comm_rate = rates.get('Commercial', 0) * 100
cust_rate = rates.get('Custom', 0) * 100

print(f"\nDocumentation Rates:")
print(f"Commercial: {comm_rate:.2f}%")
print(f"Custom: {cust_rate:.2f}%")

# 7. Visualization
plt.figure(figsize=(8, 6))
bars = plt.bar(['Commercial', 'Custom'], [comm_rate, cust_rate], color=['orange', 'skyblue'], edgecolor='black')
plt.title('Transparency Gap: Commercial vs Custom AI Documentation')
plt.ylabel('Percentage of Systems with Data Documentation (%)')
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
