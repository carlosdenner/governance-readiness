import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
import sys

# Check for dataset in current or parent directory
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(f'../{filename}'):
    filepath = f'../{filename}'
else:
    print(f"Error: {filename} not found.")
    print(f"CWD: {os.getcwd()}")
    print(f"Files in CWD: {os.listdir('.')}")
    try:
        print(f"Files in Parent: {os.listdir('..')}")
    except: 
        pass
    sys.exit(1)

print(f"Loading dataset from {filepath}...")
df = pd.read_csv(filepath, low_memory=False)

# --- 1. Prepare AIID Data ---
aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# Identify Sector column (Metadata says '78: Sector of Deployment')
# We'll look for it by name similarity to be safe
sector_cols = [c for c in aiid.columns if 'Sector' in c and 'Deployment' in c]
sector_col = sector_cols[0] if sector_cols else '78: Sector of Deployment'

# Clean and Aggregate AIID
# Remove NaN sectors
aiid = aiid.dropna(subset=[sector_col])
# Standardize names (lowercase, strip)
aiid['clean_sector'] = aiid[sector_col].astype(str).str.lower().str.strip()

sector_counts = aiid['clean_sector'].value_counts().reset_index()
sector_counts.columns = ['sector_term', 'incident_count']

print("Top AIID Sectors (cleaned):")
print(sector_counts.head())

# --- 2. Prepare EO13960 Data ---
eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Identify Topic Area (Metadata: '8: 8_topic_area')
topic_cols = [c for c in eo.columns if 'topic_area' in c.lower()]
topic_col = topic_cols[0] if topic_cols else '8: 8_topic_area'

# Governance Columns to score
gov_cols_map = {
    'Impact Assessment': [c for c in eo.columns if '52_impact_assessment' in c],
    'Bias Mitigation': [c for c in eo.columns if '62_disparity_mitigation' in c],
    'Independent Eval': [c for c in eo.columns if '55_independent_eval' in c],
    'Monitoring': [c for c in eo.columns if '56_monitor_postdeploy' in c],
    'Notice': [c for c in eo.columns if '59_ai_notice' in c],
    'Opt Out': [c for c in eo.columns if '67_opt_out' in c]
}

# Flatten list of columns found
found_gov_cols = []
for k, v in gov_cols_map.items():
    if v: found_gov_cols.append(v[0])

print(f"Governance columns used: {found_gov_cols}")

# Calculate Score
# Convert values to binary. Assume 'yes', 'true', '1' are positive.
def parse_gov_bool(val):
    s = str(val).lower().strip()
    return 1 if s in ['yes', 'true', '1', '1.0'] else 0

for col in found_gov_cols:
    eo[col + '_score'] = eo[col].apply(parse_gov_bool)

score_cols = [c + '_score' for c in found_gov_cols]
eo['gov_score'] = eo[score_cols].mean(axis=1)

# Aggregate by Topic
eo['clean_topic'] = eo[topic_col].astype(str).str.lower().str.strip()
topic_scores = eo.groupby('clean_topic')['gov_score'].mean().reset_index()

print("Top EO Topics:")
print(topic_scores.head())

# --- 3. Match Sectors and Topics ---
# Strategy: Fuzzy match or explicit mapping.
# EO Topics are US Federal specific. AIID are general.
# We will try to map common ones manually to ensure accuracy, then fallback to keyword match.

mapping_pairs = []

# Simple keyword mapping logic
for topic in topic_scores['clean_topic'].unique():
    # Keywords for this topic
    topic_words = set(w for w in topic.split() if len(w) > 3)
    
    matched_incidents = 0
    matched_sectors_list = []
    
    for _, row in sector_counts.iterrows():
        sect = row['sector_term']
        count = row['incident_count']
        
        # Check for intersection of significant words
        sect_words = set(w for w in sect.split() if len(w) > 3)
        
        # Custom overrides/synonyms
        synonyms = {
            'health': ['healthcare', 'medicine', 'medical', 'hospital'],
            'transportation': ['automotive', 'vehicle', 'driving', 'airplane', 'aviation'],
            'law enforcement': ['police', 'surveillance', 'crime', 'criminal', 'justice'],
            'finance': ['financial', 'banking', 'trading', 'credit'],
            'education': ['school', 'university', 'student', 'teaching'],
            'energy': ['power', 'grid', 'electricity', 'utility']
        }
        
        is_match = False
        if topic == sect:
            is_match = True
        elif topic_words & sect_words: # Overlap
            is_match = True
        else:
            # Check synonyms
            for k, vals in synonyms.items():
                if k in topic:
                    if any(v in sect for v in vals):
                        is_match = True
        
        if is_match:
            matched_incidents += count
            matched_sectors_list.append(sect)
            
    if matched_incidents > 0:
        mapping_pairs.append({
            'Topic': topic,
            'Governance_Score': topic_scores[topic_scores['clean_topic'] == topic]['gov_score'].values[0],
            'Incident_Count': matched_incidents,
            'Matched_Sectors': ', '.join(matched_sectors_list[:3]) # Show first few
        })

result_df = pd.DataFrame(mapping_pairs)
print("\n--- Merged Data Analysis Frame ---")
print(result_df.sort_values('Incident_Count', ascending=False))

# --- 4. Correlation Analysis ---
if len(result_df) > 3:
    corr, p_val = spearmanr(result_df['Incident_Count'], result_df['Governance_Score'])
    print(f"\nSpearman Correlation: {corr:.4f} (p-value: {p_val:.4f})")
    
    corr_p, p_val_p = pearsonr(result_df['Incident_Count'], result_df['Governance_Score'])
    print(f"Pearson Correlation: {corr_p:.4f} (p-value: {p_val_p:.4f})")

    # Plot
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=result_df, x='Incident_Count', y='Governance_Score', s=100)
    
    # Regression line
    sns.regplot(data=result_df, x='Incident_Count', y='Governance_Score', scatter=False, color='red', ci=None)

    # Labels
    for i, row in result_df.iterrows():
        plt.text(row['Incident_Count']+1, row['Governance_Score'], 
                 row['Topic'].title(), fontsize=9)

    plt.title(f'The Reality Gap: AI Incidents vs. Governance Readiness\n(Spearman r={corr:.2f}, p={p_val:.2f})')
    plt.xlabel('Reported AI Incidents (AIID)')
    plt.ylabel('Governance Readiness Score (EO 13960)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient data points for correlation.")
