import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# [debug] import sys; print(sys.version)

def normalize_sector(text):
    if pd.isna(text):
        return 'Other'
    text = str(text).lower()
    
    # Priority mapping
    if any(k in text for k in ['defense', 'security', 'military', 'border', 'justice', 'law', 'police', 'surveillance', 'intelligence', 'homeland']):
        return 'Defense & Security'
    if any(k in text for k in ['health', 'medic', 'hospital', 'care', 'hhs']):
        return 'Healthcare'
    if any(k in text for k in ['transport', 'vehicle', 'traffic', 'aviation', 'mobility', 'automotive', 'driver']):
        return 'Transportation'
    if any(k in text for k in ['financ', 'bank', 'econom', 'tax', 'insurance', 'fiscal', 'treasury']):
        return 'Finance'
    if any(k in text for k in ['educat', 'school', 'universit', 'learning', 'teach']):
        return 'Education'
    if any(k in text for k in ['labor', 'work', 'employ', 'job', 'social', 'welfare', 'housing', 'human services', 'benefit']):
        return 'Labor & Social Services'
    if any(k in text for k in ['energy', 'power', 'grid', 'utilit', 'electric']):
        return 'Energy'
    if any(k in text for k in ['agric', 'farm', 'environment', 'climate', 'weather', 'land', 'forest', 'park', 'interior', 'natural resource']):
        return 'Agriculture & Environment'
    if any(k in text for k in ['science', 'technolog', 'research', 'space', 'nasa', 'nuclear']):
        return 'Science & Tech'
    if any(k in text for k in ['commerce', 'trade', 'business', 'market', 'retail', 'consumer']):
        return 'Commerce'
    
    return 'General Gov / Other'

# 1. Load Data
print("Loading dataset...")
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Fallback for local testing if file is in current dir
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter Subsets
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

print(f"EO 13960 Records: {len(eo_df)}")
print(f"AIID Incidents: {len(aiid_df)}")

# 3. Normalize Sectors
# EO Column: '8_topic_area'. Sometimes Agency name is a good proxy if topic is missing, but we stick to topic per instructions.
# AIID Column: '78_Sector of Deployment' -> The column name in the CSV might be 'Sector of Deployment' based on previous outputs.
# Let's check available columns to be safe, searching for 'Sector' and 'topic'.

eo_col = '8_topic_area'
aiid_col = 'Sector of Deployment'

# Verify columns exist
if eo_col not in eo_df.columns:
    # Try finding it by index or similar name
    eo_col = [c for c in eo_df.columns if 'topic' in c.lower()][0]

if aiid_col not in aiid_df.columns:
    # Try finding it
    aiid_col = [c for c in aiid_df.columns if 'sector' in c.lower() and 'deployment' in c.lower()]
    if aiid_col: aiid_col = aiid_col[0]
    else: aiid_col = '78_Sector of Deployment' # Fallback to what was seen in metadata

print(f"Using EO Column: {eo_col}")
print(f"Using AIID Column: {aiid_col}")

eo_df['norm_sector'] = eo_df[eo_col].apply(normalize_sector)
aiid_df['norm_sector'] = aiid_df[aiid_col].apply(normalize_sector)

# 4. Calculate Shares
eo_counts = eo_df['norm_sector'].value_counts(normalize=True) * 100
aiid_counts = aiid_df['norm_sector'].value_counts(normalize=True) * 100

# 5. Merge
sector_stats = pd.DataFrame({'EO_Share': eo_counts, 'AIID_Share': aiid_counts}).fillna(0)

# Sort for consistency
sector_stats = sector_stats.sort_values('EO_Share', ascending=False)

print("\n--- Sector Share Comparison ---")
print(sector_stats)

# 6. Statistical Test
# Spearman Rank Correlation
corr, p_val = stats.spearmanr(sector_stats['EO_Share'], sector_stats['AIID_Share'])

print(f"\nSpearman Correlation: {corr:.4f}")
print(f"P-value: {p_val:.4f}")

# 7. Visualization
plt.figure(figsize=(10, 8))
plt.scatter(sector_stats['EO_Share'], sector_stats['AIID_Share'], color='blue', s=100)

# Add labels
for idx, row in sector_stats.iterrows():
    plt.text(row['EO_Share'] + 0.5, row['AIID_Share'], idx, fontsize=9)

# Add 45-degree line
max_val = max(sector_stats['EO_Share'].max(), sector_stats['AIID_Share'].max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Alignment')

plt.title('Risk-Investment Mismatch: Gov Use Cases vs. Real-World Incidents')
plt.xlabel('Government Investment Share (EO 13960) %')
plt.ylabel('Real-World Incident Share (AIID) %')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
