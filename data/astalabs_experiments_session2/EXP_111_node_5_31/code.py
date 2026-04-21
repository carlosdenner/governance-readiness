import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np

# [1] Load Data
filename = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filename, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# [2] Filter Source
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID incidents: {len(aiid)}")

# [3] Robust Text Concatenation
# Identify columns
potential_text_cols = ['title', 'description', 'summary', 'Alleged harmed or nearly harmed parties']
found_text_cols = [c for c in potential_text_cols if c in aiid.columns]
print(f"Using columns for text analysis: {found_text_cols}")

# Fill NaNs with empty string and ensure string type for each column individually
for col in found_text_cols:
    aiid[col] = aiid[col].fillna('').astype(str)

# Concatenate
aiid['full_text'] = aiid[found_text_cols].agg(' '.join, axis=1).str.lower()

# [4] Classification Logic

# Sector Classification
def get_sector(val):
    if not isinstance(val, str):
        return None
    val = val.lower()
    if 'health' in val or 'medic' in val or 'hospital' in val or 'doctor' in val:
        return 'Healthcare'
    if 'financ' in val or 'bank' in val or 'insurance' in val or 'invest' in val or 'trading' in val:
        return 'Financial'
    return None

# Harm Classification (Keyword-based)
def get_harm_type(text):
    if not isinstance(text, str):
        return None
    
    # Keywords for Physical Harm
    physical_keys = [
        'death', 'kill', 'dead', 'injur', 'hurt', 'physical', 'fatal', 
        'accident', 'crash', 'collision', 'burn', 'medical', 'patient', 
        'hospital', 'surgery', 'pain', 'assault', 'hit', 'struck'
    ]
    
    # Keywords for Economic Harm
    economic_keys = [
        'money', 'financ', 'dollar', 'cost', 'fund', 'bank', 'credit', 
        'fraud', 'scam', 'loss', 'market', 'trade', 'economic', 'price', 
        'fee', 'charge', 'wealth', 'asset', 'theft', 'steal', 'embezzle'
    ]
    
    # Count occurrences
    p_count = sum(1 for k in physical_keys if k in text)
    e_count = sum(1 for k in economic_keys if k in text)
    
    if p_count > 0 and e_count == 0:
        return 'Physical'
    if e_count > 0 and p_count == 0:
        return 'Economic'
    if p_count > e_count:
        return 'Physical'
    if e_count > p_count:
        return 'Economic'
    
    # If tied or neither, we can't definitively classify for this binary test
    return None

# Apply Classification
aiid['analyzed_sector'] = aiid['Sector of Deployment'].fillna('').apply(get_sector)
aiid['analyzed_harm'] = aiid['full_text'].apply(get_harm_type)

# [5] Filter Data
analysis_df = aiid.dropna(subset=['analyzed_sector', 'analyzed_harm']).copy()

print(f"\nIncidents after filtering for (Healthcare/Financial) and (Physical/Economic): {len(analysis_df)}")
print("Distribution:")
print(analysis_df.groupby(['analyzed_sector', 'analyzed_harm']).size())

# [6] Statistical Test and Visualization
if len(analysis_df) > 5:
    # Contingency Table
    contingency_table = pd.crosstab(analysis_df['analyzed_sector'], analysis_df['analyzed_harm'])
    print("\nContingency Table:")
    print(contingency_table)
    
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    props = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    ax = props.plot(kind='bar', stacked=True, color=['#FF9999', '#66B2FF'], edgecolor='black')
    
    plt.title('Proportion of Harm Types by Sector (Healthcare vs. Financial)')
    plt.xlabel('Sector')
    plt.ylabel('Proportion')
    plt.legend(title='Harm Type (Derived)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', label_type='center')
        
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data for valid analysis.")
