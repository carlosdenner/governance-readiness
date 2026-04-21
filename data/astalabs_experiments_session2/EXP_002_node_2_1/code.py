import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os

# 1. Load Data
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, f'../{filename}']
file_path = None

for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if not file_path:
    file_path = filename

print(f"Loading data from: {file_path}")

try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# 2. Filter for aiid_incidents
df_incidents = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents shape: {df_incidents.shape}")

# 3. Define Proxies for Physical and Psychological Harm
# Based on previous exploration, specific 'Physical'/'Psychological' labels are not in a single column.
# We hypothesize:
# - 'Physical' corresponds to 'Tangible Harm' being present.
# - 'Psychological' corresponds to 'Special Interest Intangible Harm' being 'yes'.

# Check values again to be sure
print("Tangible Harm values:", df_incidents['Tangible Harm'].dropna().unique())
if 'Special Interest Intangible Harm' in df_incidents.columns:
    print("Special Interest Intangible Harm values:", df_incidents['Special Interest Intangible Harm'].dropna().unique())

# Function to tag rows
def tag_harm_type(row):
    tags = []
    
    # Physical / Tangible
    # We consider it Physical if tangible harm definitively occurred or there was an imminent risk.
    tangible = str(row['Tangible Harm']).lower()
    if 'definitively occurred' in tangible or 'imminent risk' in tangible:
        tags.append('Physical')
        
    # Psychological / Intangible
    # We use the Special Interest Intangible Harm column
    intangible = str(row.get('Special Interest Intangible Harm', '')).lower()
    if intangible == 'yes':
        tags.append('Psychological')
    
    return tags

# Apply tagging
df_incidents['Derived_Harm_Tags'] = df_incidents.apply(tag_harm_type, axis=1)

# Explode tags so we can analyze distribution per tag
df_exploded = df_incidents.explode('Derived_Harm_Tags')
df_analysis = df_exploded.dropna(subset=['Derived_Harm_Tags'])

print(f"Rows with derived tags: {len(df_analysis)}")
print("Tag counts:\n", df_analysis['Derived_Harm_Tags'].value_counts())

if df_analysis.empty:
    print("No tags derived. Cannot proceed.")
    exit(0)

# 4. Clean Sector Column
sector_col = 'Sector of Deployment'

# Helper to explode sectors if they are lists
def clean_and_explode(dataframe, column):
    s = dataframe[column].astype(str).str.split(',')
    dataframe = dataframe.assign(**{column: s}).explode(column)
    dataframe[column] = dataframe[column].str.strip()
    return dataframe

df_analysis = clean_and_explode(df_analysis, sector_col)
df_analysis = df_analysis[~df_analysis[sector_col].isin(['', 'nan', 'NaN'])]

# 5. Analysis: Crosstab and Entropy
ct = pd.crosstab(df_analysis['Derived_Harm_Tags'], df_analysis[sector_col])
probs = ct.div(ct.sum(axis=1), axis=0)

entropy_scores = probs.apply(lambda x: scipy.stats.entropy(x, base=2), axis=1)

print("\n--- Entropy Scores (Lower = More Localized) ---")
print(entropy_scores.sort_values())

# 6. Validate Hypothesis Sectors
# Hypothesis:
# Physical -> Transportation, Industrial (Manufacturing?)
# Psychological -> Social Media, Healthcare, Entertainment

target_map = {
    'Physical': ['transportation', 'industrial', 'manufacturing'],
    'Psychological': ['social media', 'healthcare', 'entertainment']
}

print("\n--- Sector Proportions by Harm Type ---")
for harm in ['Physical', 'Psychological']:
    if harm in probs.index:
        print(f"\n{harm} Harm:")
        # Show top 5
        top = probs.loc[harm].sort_values(ascending=False).head(5)
        for s, p in top.items():
            print(f"  {s}: {p:.1%}")
            
        # Check specific hypothesis targets
        print(f"  > Hypothesis Check:")
        targets = target_map.get(harm, [])
        for t in targets:
            # Find matching sector keys
            matches = [k for k in probs.columns if t in k.lower()]
            for m in matches:
                print(f"    {m}: {probs.loc[harm, m]:.1%}")

# 7. Visualization
top_sectors_idx = ct.sum(axis=0).sort_values(ascending=False).head(10).index
plot_data = probs[top_sectors_idx]

if not plot_data.empty:
    ax = plot_data.T.plot(kind='bar', figsize=(10, 6))
    plt.title('Sector Distribution: Physical (Tangible) vs Psychological (Intangible)')
    plt.ylabel('Proportion')
    plt.xlabel('Sector')
    plt.legend(title='Derived Harm Type')
    plt.tight_layout()
    plt.show()
