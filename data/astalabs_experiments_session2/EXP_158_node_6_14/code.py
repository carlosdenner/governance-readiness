import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# Create a text corpus for classification from potential text columns
# We prioritize columns known to exist or likely to contain descriptive text
text_cols = ['title', 'description', 'summary', 'reports', 'Sector of Deployment', 'Infrastructure Sectors', 'Harm Domain', 'Tangible Harm', 'Alleged harmed or nearly harmed parties']

# Combine available columns into a single string for keyword searching
df_aiid['text_corpus'] = ''
for col in text_cols:
    if col in df_aiid.columns:
        df_aiid['text_corpus'] += ' ' + df_aiid[col].fillna('').astype(str)

df_aiid['text_corpus'] = df_aiid['text_corpus'].str.lower()

# --- 1. Classify Sectors ---
sector_map = {
    'Financial': ['financ', 'bank', 'credit', 'loan', 'insurance', 'trading', 'mortgage', 'lending', 'crypto'],
    'Transportation': ['transport', 'auto', 'vehicle', 'car', 'aviation', 'flight', 'drone', 'driverless', 'tesla', 'uber', 'collision', 'autopilot', 'self-driving']
}

def classify_sector(text):
    scores = {cat: 0 for cat in sector_map}
    for cat, keywords in sector_map.items():
        for k in keywords:
            if k in text:
                scores[cat] += 1
    
    # Return the category with the highest non-zero score
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return None

df_aiid['derived_sector'] = df_aiid['text_corpus'].apply(classify_sector)

# --- 2. Classify Harms ---
harm_map = {
    'Physical': ['death', 'kill', 'injur', 'hurt', 'accident', 'crash', 'safety', 'physical', 'died', 'fatal', 'bodily'],
    'Non-Physical': ['economic', 'money', 'financ', 'cost', 'discrimin', 'bias', 'racist', 'sexist', 'privacy', 'surveillance', 'reputation', 'credit score', 'denied', 'unfair']
}

def classify_harm(text):
    # Priority check: If 'death' or 'injury' is explicitly mentioned, it's Physical (safety critical)
    # However, we'll use a scoring system to be robust
    scores = {cat: 0 for cat in harm_map}
    for cat, keywords in harm_map.items():
        for k in keywords:
            if k in text:
                scores[cat] += 1
    
    # Heuristic: Physical harm usually implies safety incidents which are distinct from pure economic/bias
    # If both present, usually the Physical aspect is the 'incident' trigger (e.g., crash)
    if scores['Physical'] > 0:
        return 'Physical'
    elif scores['Non-Physical'] > 0:
        return 'Non-Physical'
    return None

df_aiid['derived_harm'] = df_aiid['text_corpus'].apply(classify_harm)

# --- 3. Analysis ---
# Filter for rows where both Sector and Harm were identified
df_analysis = df_aiid.dropna(subset=['derived_sector', 'derived_harm'])

print(f"Total AIID Incidents: {len(df_aiid)}")
print(f"Incidents with identified Sector & Harm: {len(df_analysis)}")
print("Sector Breakdown in Analysis Set:")
print(df_analysis['derived_sector'].value_counts())

# Contingency Table
contingency = pd.crosstab(df_analysis['derived_sector'], df_analysis['derived_harm'])
print("\nContingency Table (Sector vs Harm Type):")
print(contingency)

# Statistical Test
if contingency.size >= 4:
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    
    # Calculate Percentages
    row_props = pd.crosstab(df_analysis['derived_sector'], df_analysis['derived_harm'], normalize='index') * 100
    print("\nHarm Type Distribution by Sector (%):")
    print(row_props.round(2))
    
    # Check Hypothesis
    # Hypothesis: Financial -> Non-Physical (Economic), Transportation -> Physical
    fin_non_phys = row_props.loc['Financial', 'Non-Physical'] if 'Financial' in row_props.index and 'Non-Physical' in row_props.columns else 0
    trans_phys = row_props.loc['Transportation', 'Physical'] if 'Transportation' in row_props.index and 'Physical' in row_props.columns else 0
    
    print(f"\nFinancial Incidents causing Non-Physical Harm: {fin_non_phys:.1f}%")
    print(f"Transportation Incidents causing Physical Harm: {trans_phys:.1f}%")
    
    # Plot
    try:
        row_props.plot(kind='bar', stacked=True, color=['orange', 'red'], alpha=0.7, figsize=(8, 6))
        plt.title('Harm Fingerprints: Physical vs Non-Physical Harm by Sector')
        plt.xlabel('Sector')
        plt.ylabel('Percentage of Incidents')
        plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")
else:
    print("Insufficient data for statistical test.")
