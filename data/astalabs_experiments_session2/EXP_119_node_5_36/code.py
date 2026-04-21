import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

print("Starting experiment...")

# 1. Load Data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
df_incidents = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded {len(df_incidents)} incidents from AIID.")

# 2. Define Groups based on 'Tangible Harm' strings found in debug
# Physical: Actual harm or imminent risk of harm
# Intangible: Explicitly labeled as 'no tangible harm' (implying other types of harm like allocative/societal)

def categorize_harm(val):
    if pd.isna(val):
        return None
    s = str(val).lower()
    
    # Physical Keywords based on debug output
    if "tangible harm definitively occurred" in s:
        return "Physical"
    if "imminent risk" in s and "did occur" in s:
        return "Physical"
        
    # Intangible Keywords
    if "no tangible harm" in s:
        return "Intangible"
        
    return None

df_incidents['Harm_Group'] = df_incidents['Tangible Harm'].apply(categorize_harm)

# 3. Define Technical Cause
# 1 if 'Known AI Technical Failure' is populated, 0 otherwise
def has_tech_cause(val):
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s in ["", "nan", "None", "[]"]:
        return 0
    return 1

df_incidents['Has_Technical_Cause'] = df_incidents['Known AI Technical Failure'].apply(has_tech_cause)

# 4. Filter for Analysis
df_analysis = df_incidents.dropna(subset=['Harm_Group']).copy()

print("\n--- Analysis Groups ---@")
print(df_analysis['Harm_Group'].value_counts())

# 5. Statistical Test
contingency = pd.crosstab(df_analysis['Harm_Group'], df_analysis['Has_Technical_Cause'])
print("\n--- Contingency Table (0=No Tech Cause, 1=Has Tech Cause) ---")
print(contingency)

if contingency.shape == (2, 2):
    # Chi-Square Test
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:\n  Chi2 Statistic: {chi2:.4f}\n  P-value: {p:.5f}")
    
    # Calculate Attribution Rates
    rates = df_analysis.groupby('Harm_Group')['Has_Technical_Cause'].mean()
    print("\n--- Attribution Rates (Proportion with Known Cause) ---")
    print(rates)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    # Color: Red for Physical, Blue for Intangible
    colors = ['#3498db' if x == 'Intangible' else '#e74c3c' for x in rates.index]
    bars = plt.bar(rates.index, rates.values, color=colors, alpha=0.8)
    
    plt.title('"Causal Clarity": Technical Attribution by Harm Type')
    plt.ylabel('Proportion of Incidents with Identified Technical Failure')
    plt.ylim(0, max(rates.values) * 1.2 if max(rates.values) > 0 else 1.0)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, 
                 f'{height:.1%}', ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

else:
    print("\nInsufficient data for 2x2 Chi-Square test.")
    print("Rows with defined Harm Group:", len(df_analysis))
