import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# [debug] Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# --- ROBUST COLUMN DISCOVERY ---
# We need to find the column that actually contains the text describing the harm (Physical/Economic).
# We also confirm the sector column.

def count_matches(series, keywords):
    # Convert to string, lowercase, count rows containing any keyword
    return series.astype(str).str.lower().apply(lambda x: any(k in x for k in keywords)).sum()

# Define keywords
sector_keywords = ['healthcare', 'hospital', 'medical', 'finance', 'banking', 'insurance']
harm_keywords = ['physical', 'injury', 'death', 'safety', 'economic', 'monetary', 'financial loss']

# Scan text columns
sector_scores = {}
harm_scores = {}

# Only scan object columns to save time/errors
text_cols = aiid_df.select_dtypes(include=['object']).columns

for col in text_cols:
    # Skip system columns like ids
    if 'id' in col.lower() and 'description' not in col.lower():
        continue
        
    # Score for Sector
    sector_scores[col] = count_matches(aiid_df[col], sector_keywords)
    
    # Score for Harm
    harm_scores[col] = count_matches(aiid_df[col], harm_keywords)

# Select best columns
# For Sector: Prefer columns with 'sector' in name if scores are comparable
best_sector_col = max(sector_scores, key=sector_scores.get)
print(f"Top Sector Column Candidate: {best_sector_col} (Matches: {sector_scores[best_sector_col]})")

# For Harm: description or summary is usually best for unstructured extraction if structured fails
# We check the structured ones first, but if they score low, we take description.
best_harm_col = max(harm_scores, key=harm_scores.get)
print(f"Top Harm Column Candidate: {best_harm_col} (Matches: {harm_scores[best_harm_col]})")

# Force check: if 'description' has more info, use it.
if 'description' in harm_scores and harm_scores['description'] > harm_scores.get(best_harm_col, 0):
    best_harm_col = 'description'

print(f"\nSelected Columns -> Sector: '{best_sector_col}', Harm Source: '{best_harm_col}'")

# --- MAPPING FUNCTIONS ---

def get_sector_group(row):
    text = str(row[best_sector_col]).lower()
    if any(k in text for k in ['health', 'medic', 'hospital']):
        return 'Healthcare'
    if any(k in text for k in ['financ', 'bank', 'insurance']):
        return 'Financial'
    return None

def get_harm_group(row):
    text = str(row[best_harm_col]).lower()
    
    # Physical Harm Indicators
    physical_keys = ['physical', 'injury', 'death', 'safety', 'kill', 'hurt', 'bodily', 'violence']
    if any(k in text for k in physical_keys):
        return 'Physical'
        
    # Economic Harm Indicators
    economic_keys = ['economic', 'monetary', 'money', 'loss', 'fraud', 'theft', 'scam', 'credit']
    # Note: 'financial' is skipped here if using a shared column to avoid confounding with sector name,
    # unless the context is clear. We'll include it but be careful.
    if any(k in text for k in economic_keys) or ('financial' in text and 'sector' not in text):
        return 'Economic'
        
    return 'Other'

# Apply Mappings
aiid_df['Sector_Group'] = aiid_df.apply(get_sector_group, axis=1)
aiid_df['Harm_Group'] = aiid_df.apply(get_harm_group, axis=1)

# Filter for analysis
analysis_df = aiid_df.dropna(subset=['Sector_Group'])

# --- STATISTICAL ANALYSIS ---
contingency_table = pd.crosstab(analysis_df['Sector_Group'], analysis_df['Harm_Group'])
print("\n--- Contingency Table (Sector vs Derived Harm Type) ---")
print(contingency_table)

if contingency_table.empty or contingency_table.values.sum() == 0:
    print("No matching data found.")
else:
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Statistically Significant (Reject H0)")
    else:
        print("Result: Not Significant (Fail to reject H0)")

    # Visualization
    plot_data = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    ax = plot_data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title(f'Harm Distribution by Sector\n(Source: {best_harm_col})')
    plt.ylabel('Percentage')
    plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
    plt.show()