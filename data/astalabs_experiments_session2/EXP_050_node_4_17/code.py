import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import re
import sys

# 1. Load Dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID Incidents: {len(aiid_df)}")

# 2. Define Columns
deployer_col = 'Alleged deployer of AI system'
harm_dist_col = 'Harm Distribution Basis'
tangible_col = 'Tangible Harm'

# 3. Clean and Categorize Sector
def categorize_sector(val):
    if pd.isna(val):
        return 'Unknown'
    # Clean JSON artifacts
    val_clean = re.sub(r'[\[\]"\']', '', str(val)).lower()
    
    # Explicit Private Entities (Top frequency check)
    private_entities = [
        'tesla', 'google', 'openai', 'facebook', 'amazon', 'meta', 'microsoft', 
        'cruise', 'waymo', 'uber', 'xai', 'tiktok', 'youtube', 'apple', 'twitter',
        'snapchat', 'instagram', 'whatsapp', 'linkedin', 'salesforce', 'ibm', 'intel',
        'adobe', 'oracle', 'nvidia', 'palantir', 'deepmind', 'stability ai', 'midjourney'
    ]
    
    # General Private Keywords
    private_keywords = [
        'inc', 'corp', 'llc', 'ltd', 'company', 'technologies', 'systems', 
        'solutions', 'group', 'motors', 'airlines', 'bank', 'entertainment'
    ]
    
    # Public Keywords
    public_keywords = [
        'police', 'government', 'dept', 'department', 'ministry', 'agency', 
        'commission', 'authority', 'council', 'state', 'city', 'county', 
        'federal', 'national', 'bureau', 'sheriff', 'nhs', 'army', 'navy', 
        'air force', 'dhs', 'fbi', 'cia', 'school', 'university', 'college', 
        'court', 'judge', 'municipality', 'parliament', 'congress'
    ]
    
    # Classification Logic
    if any(e == val_clean or e in val_clean.split('-') for e in private_entities):
        return 'Private'
    if any(k in val_clean for k in private_keywords):
        return 'Private'
    if any(k in val_clean for k in public_keywords):
        return 'Public'
        
    return 'Other'

# 4. Categorize Harm Type
def categorize_harm(row):
    dist_basis = str(row.get(harm_dist_col, '')).lower()
    tangible = str(row.get(tangible_col, '')).lower()
    
    # Allocative Signal: Valid entry in 'Harm Distribution Basis'
    # Exclude: 'none', 'unclear', 'nan', empty strings
    if dist_basis not in ['none', 'unclear', 'nan', '']:
        return 'Allocative'
    
    # Physical Signal: 'Tangible Harm' explicitly indicates occurrence
    if 'definitively occurred' in tangible or 'imminent risk' in tangible:
        return 'Physical'
        
    return 'Other'

# Apply Categorization
aiid_df['Sector_Category'] = aiid_df[deployer_col].apply(categorize_sector)
aiid_df['Harm_Category'] = aiid_df.apply(categorize_harm, axis=1)

# 5. Filter for Analysis
analysis_df = aiid_df[
    (aiid_df['Sector_Category'].isin(['Public', 'Private'])) & 
    (aiid_df['Harm_Category'].isin(['Allocative', 'Physical']))
].copy()

print(f"Records for Analysis: {len(analysis_df)}")
print("\n--- Counts by Sector ---")
print(analysis_df['Sector_Category'].value_counts())
print("\n--- Counts by Harm Type ---")
print(analysis_df['Harm_Category'].value_counts())

if len(analysis_df) < 5:
    print("Insufficient data for statistical testing.")
    sys.exit(0)

# 6. Statistical Test (Chi-Square)
contingency_table = pd.crosstab(analysis_df['Sector_Category'], analysis_df['Harm_Category'])
print("\n--- Contingency Table ---")
print(contingency_table)

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n--- Chi-Square Results ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

if p < 0.05:
    print("Conclusion: Significant relationship found between Sector and Harm Type.")
else:
    print("Conclusion: No significant relationship found.")

# 7. Visualization
# Normalize to get proportions
props = contingency_table.div(contingency_table.sum(axis=1), axis=0)

plt.figure(figsize=(10, 6))
# Plot stacked bar
ax = props.plot(kind='bar', stacked=True, color=['#d62728', '#1f77b4'], ax=plt.gca())

plt.title('Proportion of Allocative vs. Physical Harms by Sector')
plt.ylabel('Proportion')
plt.xlabel('Deployer Sector')
plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()

# Annotate bars
for c in ax.containers:
    # Only label non-zero segments
    labels = [f'{v.get_height():.2f}' if v.get_height() > 0.01 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold')

plt.show()
