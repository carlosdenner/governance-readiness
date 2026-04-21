import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Build Tactic ID -> Name Mapping
# Try to find mapping in cross_taxonomy_map
map_df = df[df['source_table'] == 'cross_taxonomy_map']

# Extract potential mappings from source_id -> source_label
# Assuming ATLAS IDs are in source_id (e.g., AML.TAxxxx)
tactic_map = {}

# Check source columns
for idx, row in map_df.iterrows():
    if isinstance(row['source_id'], str) and 'AML.TA' in row['source_id'] and isinstance(row['source_label'], str):
        tactic_map[row['source_id']] = row['source_label']
    # Check target columns just in case
    if isinstance(row['target_id'], str) and 'AML.TA' in row['target_id'] and isinstance(row['target_label'], str):
        tactic_map[row['target_id']] = row['target_label']

# If map is empty or small, try to extract from other columns if they exist (fallback)
if not tactic_map:
    # Try to find any column combination in the whole df that links id to name
    # This is a bit expensive so we rely on the map_df first.
    pass

# Hardcoded fallback if mapping is missing (Based on MITRE ATLAS standard if necessary, but prefer data)
# Just in case the data is missing the labels.
if 'AML.TA0005' not in tactic_map:
    # partial fallback for key items mentioned in prompt if not found
    tactic_map.update({
        'AML.TA0000': 'Reconnaissance',
        'AML.TA0001': 'Resource Development',
        'AML.TA0002': 'Initial Access',
        'AML.TA0003': 'ML Model Access',
        'AML.TA0004': 'ML Attack Staging',
        'AML.TA0005': 'Defense Evasion',  # Often just Evasion
        'AML.TA0006': 'Discovery',
        'AML.TA0007': 'Persistence',
        'AML.TA0008': 'Privilege Escalation',
        'AML.TA0009': 'Lateral Movement',
        'AML.TA0010': 'Exfiltration', # Key for hypothesis
        'AML.TA0011': 'Impact',
        'AML.TA0043': 'Reconnaissance', # 2024 updates sometimes change IDs, but stick to data
    })

print(f"Tactic Mapping (First 5): {dict(list(tactic_map.items())[:5])}")

# 3. Process Incident Coding
incidents = df[df['source_table'] == 'step3_incident_coding'].copy()

# Calculate Competency Gaps Count
# Assuming semicolon delimited. Empty string/NaN is 0.
def count_gaps(val):
    if pd.isna(val) or val == '':
        return 0
    return str(val).count(';') + 1

incidents['gap_count'] = incidents['competency_gaps'].apply(count_gaps)

# Process Tactics
# Split tactics_used by '; ' or ';'
incidents['tactics_list'] = incidents['tactics_used'].astype(str).apply(lambda x: [t.strip() for t in x.replace('; ', ';').split(';') if 'AML.TA' in t])

# Explode
exploded = incidents.explode('tactics_list')

# Map Tactic Names
exploded['tactic_name'] = exploded['tactics_list'].map(tactic_map)

# Handle unmapped tactics (use ID if Name not found)
exploded['tactic_name'] = exploded['tactic_name'].fillna(exploded['tactics_list'])

# Consolidate Names (e.g. 'Defense Evasion' -> 'Evasion' to match prompt terminology if needed)
exploded['tactic_group'] = exploded['tactic_name'].apply(lambda x: 'Evasion' if 'Evasion' in str(x) else x)
exploded['tactic_group'] = exploded['tactic_group'].apply(lambda x: 'Exfiltration' if 'Exfiltration' in str(x) else x)

# 4. Analyze
# Group by Tactic Group and calculate mean gap_count
grouped = exploded.groupby('tactic_group')['gap_count'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)

print("\nAverage Competency Gaps by Tactic:")
print(grouped)

# Identify 'Evasion' vs 'Exfiltration'
target_tactics = ['Evasion', 'Exfiltration']
subset = grouped[grouped.index.isin(target_tactics)]
print("\nComparison for Hypothesis:")
print(subset)

# 5. Plot
plt.figure(figsize=(10, 6))
# Filter for meaningful groups (N > 2)
plot_data = grouped[grouped['count'] > 2]

plt.bar(plot_data.index, plot_data['mean'], yerr=plot_data['std'].fillna(0), capsize=5)
plt.title('Average Governance Gaps per Adversarial Tactic')
plt.ylabel('Avg. Competency Gaps per Incident')
plt.xlabel('Tactic')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 6. Conclusion
highest_tactic = grouped.index[0]
print(f"\nThe tactic with the highest mean gaps is: {highest_tactic}")
