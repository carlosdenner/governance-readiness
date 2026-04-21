import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re

# [debug] Step 1: Load Data
print("Loading dataset...")
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents found: {len(aiid)}")

# Helper function to clean text
def clean_text(x):
    return str(x).lower() if pd.notnull(x) else ""

# Combine text fields for fallback classification
aiid['text_content'] = aiid['title'].apply(clean_text) + " " + \
                       aiid['description'].apply(clean_text) + " " + \
                       aiid['summary'].apply(clean_text)

# --- 1. Autonomy Classification ---
# Strategy: Check structured column '81_Autonomy Level', fallback to keyword search

high_autonomy_keywords = [
    'autonomous', 'self-driving', 'driverless', 'autopilot', 'robot', 'robotic', 
    'drone', 'uav', 'unmanned', 'tesla', 'waymo', 'cruise', 'uber atg'
]

def classify_autonomy(row):
    # Try structured column first
    val = str(row.get('81_Autonomy Level', '')).lower()
    if 'high' in val or 'autonomous' in val:
        return 'High Autonomy'
    if 'low' in val or 'human' in val:
        return 'Low Autonomy'
    
    # Fallback to text
    text = row['text_content']
    if any(k in text for k in high_autonomy_keywords):
        return 'High Autonomy'
    return 'Low Autonomy' # Default bucket for non-physical/software agents

aiid['Autonomy_Class'] = aiid.apply(classify_autonomy, axis=1)

# --- 2. Harm Classification ---
# Strategy: Check structured '73_Harm Domain'/'74_Tangible Harm', fallback to keywords

physical_harm_keywords = [
    'death', 'dead', 'kill', 'injury', 'injured', 'crash', 'collision', 
    'accident', 'hurt', 'physical', 'safety', 'burn', 'fracture', 'fatality'
]

def classify_harm(row):
    # Try structured columns
    h_domain = str(row.get('73_Harm Domain', '')).lower()
    t_harm = str(row.get('74_Tangible Harm', '')).lower()
    combined_struct = h_domain + " " + t_harm
    
    if 'physical' in combined_struct or 'safety' in combined_struct or 'life' in combined_struct:
        return 'Physical Harm'
    if 'economic' in combined_struct or 'opportunity' in combined_struct or 'reputation' in combined_struct:
        return 'Non-Physical Harm'
        
    # Fallback to text
    text = row['text_content']
    if any(k in text for k in physical_harm_keywords):
        return 'Physical Harm'
    return 'Non-Physical Harm'

aiid['Harm_Class'] = aiid.apply(classify_harm, axis=1)

# --- 3. Analysis ---
print("\n--- Classification Results ---")
print(aiid[['Autonomy_Class', 'Harm_Class']].value_counts())

# Contingency Table
contingency = pd.crosstab(aiid['Autonomy_Class'], aiid['Harm_Class'])
print("\n--- Contingency Table ---")
print(contingency)

# Chi-Square Test
chi2, p, dof, ex = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4e}")

# Visualization
plt.figure(figsize=(10, 6))
# Normalize by row to show probabilities
props = contingency.div(contingency.sum(axis=1), axis=0)
props.plot(kind='bar', stacked=True, color=['orange', 'red'], alpha=0.8)
plt.title('Harm Type Distribution by Autonomy Level')
plt.ylabel('Proportion of Incidents')
plt.xlabel('Autonomy Level')
plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nRESULT: Statistically significant relationship found.")
    high_phys = props.loc['High Autonomy', 'Physical Harm'] if 'High Autonomy' in props.index and 'Physical Harm' in props.columns else 0
    low_phys = props.loc['Low Autonomy', 'Physical Harm'] if 'Low Autonomy' in props.index and 'Physical Harm' in props.columns else 0
    print(f"High Autonomy systems resulted in Physical Harm in {high_phys:.1%} of cases.")
    print(f"Low Autonomy systems resulted in Physical Harm in {low_phys:.1%} of cases.")
else:
    print("\nRESULT: No statistically significant relationship found.")