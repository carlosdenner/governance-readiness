import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        exit(1)

# Filter for AIID incidents
df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded AIID Incidents: {len(df_aiid)} rows")

# Define exact mappings based on debug findings
autonomy_map = {
    'Autonomy1': 'Low',
    'Autonomy2': 'Medium',
    'Autonomy3': 'High'
}

harm_map = {
    'AI tangible harm issue': 'Minor',
    'AI tangible harm near-miss': 'Moderate',
    'AI tangible harm event': 'Severe'
}

# Apply mappings
# We use the column names identified in previous steps directly or dynamically if needed, 
# but 'Autonomy Level' and 'AI Harm Level' were confirmed.
autonomy_col = 'Autonomy Level'
harm_col = 'AI Harm Level'

if autonomy_col not in df_aiid.columns or harm_col not in df_aiid.columns:
    print("Column names mismatch. Checking columns...")
    # Fallback to dynamic search just in case
    autonomy_col = next((c for c in df_aiid.columns if 'autonomy' in c.lower() and 'level' in c.lower()), autonomy_col)
    harm_col = next((c for c in df_aiid.columns if 'harm' in c.lower() and 'level' in c.lower()), harm_col)

# Create clean columns
df_aiid['Autonomy_Clean'] = df_aiid[autonomy_col].map(autonomy_map)
df_aiid['Harm_Clean'] = df_aiid[harm_col].map(harm_map)

# Drop rows that didn't map (unclear, none, nan)
df_clean = df_aiid.dropna(subset=['Autonomy_Clean', 'Harm_Clean']).copy()

print(f"Data points after cleaning: {len(df_clean)}")
print("Autonomy Distribution:\n", df_clean['Autonomy_Clean'].value_counts())
print("Harm Distribution:\n", df_clean['Harm_Clean'].value_counts())

if len(df_clean) > 0:
    # Define Order
    autonomy_order = ['Low', 'Medium', 'High']
    harm_order = ['Minor', 'Moderate', 'Severe']
    
    # Contingency Table
    ct = pd.crosstab(df_clean['Autonomy_Clean'], df_clean['Harm_Clean'])
    ct = ct.reindex(index=autonomy_order, columns=harm_order).fillna(0)
    
    print("\nContingency Table:\n", ct)
    
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}, p-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Statistically significant relationship found between Autonomy and Harm Severity.")
    else:
        print("Result: No statistically significant relationship found.")
        
    # Calculate Row Percentages for easier interpretation
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print("\nRow Percentages (%):\n", ct_pct.round(1))
    
    # Visualization: Stacked Bar Chart
    plt.figure(figsize=(10, 6))
    # Normalize to 1.0 for stacked bar
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    
    # Plot
    ct_norm.plot(kind='bar', stacked=True, colormap='Reds', edgecolor='black', ax=plt.gca())
    
    plt.title('Harm Severity Distribution by AI Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Proportion of Incidents')
    plt.legend(title='Harm Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("No valid data available for analysis.")
