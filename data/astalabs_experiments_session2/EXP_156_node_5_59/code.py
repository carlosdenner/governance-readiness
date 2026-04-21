import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, chi2_contingency

# [debug]
print("Starting experiment: Malice is Intangible")

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    print(f"Dataset loaded. Shape: {df.shape}")
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        print(f"Dataset loaded (local). Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Filter subsets
df_atlas = df[df['source_table'] == 'atlas_cases'].copy()
df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()

print(f"ATLAS cases: {len(df_atlas)}")
print(f"AIID incidents: {len(df_aiid)}")

# identify text columns for mining
# ATLAS usually has 'summary' or 'description'
# AIID usually has 'description' or 'summary'

# Check available columns for ATLAS
atlas_text_cols = [c for c in df_atlas.columns if df_atlas[c].notna().any()]
# Prefer 'summary', then 'description', then 'name'
atlas_col = 'summary' if 'summary' in atlas_text_cols else 'description' if 'description' in atlas_text_cols else 'name'

# Check available columns for AIID
aiid_text_cols = [c for c in df_aiid.columns if df_aiid[c].notna().any()]
aiid_col = 'description' if 'description' in aiid_text_cols else 'summary' if 'summary' in aiid_text_cols else 'title'

print(f"Using column '{atlas_col}' for ATLAS text mining.")
print(f"Using column '{aiid_col}' for AIID text mining.")

# Keywords for Physical Harm
physical_keywords = [
    'death', 'dead', 'die', 'kill', 'fatality', 'fatal',
    'injury', 'injure', 'hurt', 'wound',
    'crash', 'collision', 'accident',
    'physical harm', 'bodily', 'safety',
    'destroy', 'explosion', 'fire', 'burn'
]

def check_physical(text):
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    for kw in physical_keywords:
        # Simple substring match, could be improved with regex boundary but sufficient for broad classification
        if kw in text_lower:
            return True
    return False

# Apply classification
df_atlas['is_physical'] = df_atlas[atlas_col].apply(check_physical)
df_aiid['is_physical'] = df_aiid[aiid_col].apply(check_physical)

# Aggregation
atlas_physical_count = df_atlas['is_physical'].sum()
atlas_total = len(df_atlas)
atlas_rate = atlas_physical_count / atlas_total if atlas_total > 0 else 0

aiid_physical_count = df_aiid['is_physical'].sum()
aiid_total = len(df_aiid)
aiid_rate = aiid_physical_count / aiid_total if aiid_total > 0 else 0

print(f"\nATLAS (Adversarial): {atlas_physical_count} / {atlas_total} ({atlas_rate:.1%}) physical harm incidents.")
print(f"AIID (General): {aiid_physical_count} / {aiid_total} ({aiid_rate:.1%}) physical harm incidents.")

# Statistical Test
# Contingency Table:
#              Physical | Non-Physical
# Adversarial (ATLAS) |       a  |      b
# General (AIID)      |       c  |      d

a = atlas_physical_count
b = atlas_total - atlas_physical_count
c = aiid_physical_count
d = aiid_total - aiid_physical_count

contingency_table = [[a, b], [c, d]]
print(f"\nContingency Table:\n{contingency_table}")

# Fisher's Exact Test is appropriate for small sample sizes (ATLAS has ~52)
odds_ratio, p_value = fisher_exact(contingency_table, alternative='less') 
# alternative='less' tests if ATLAS is LESS likely to have physical harm than AIID

print(f"Fisher's Exact Test p-value: {p_value:.4f}")
print(f"Odds Ratio: {odds_ratio:.4f}")

if p_value < 0.05:
    print("Result: Statistically Significant. Adversarial attacks are less likely to cause physical harm.")
else:
    print("Result: Not Statistically Significant.")

# Visualization
labels = ['Adversarial (ATLAS)', 'General (AIID)']
physical_rates = [atlas_rate * 100, aiid_rate * 100]
non_physical_rates = [100 - x for x in physical_rates]

fig, ax = plt.subplots(figsize=(8, 6))

width = 0.5
x = np.arange(len(labels))

# Stacked bar chart
p1 = ax.bar(x, physical_rates, width, label='Physical Harm', color='#d62728', alpha=0.8)
p2 = ax.bar(x, non_physical_rates, width, bottom=physical_rates, label='Non-Physical / Other', color='#1f77b4', alpha=0.8)

ax.set_ylabel('Percentage of Incidents')
ax.set_title(f'Physical Harm Rate: Adversarial vs General AI Failures\n(p={p_value:.3f})')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add percentage labels
for i, rect in enumerate(p1):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height / 2.,
            f'{height:.1f}%',
            ha='center', va='center', color='white', fontweight='bold')

for i, rect in enumerate(p2):
    height = rect.get_height()
    y_pos = physical_rates[i] + height / 2.
    ax.text(rect.get_x() + rect.get_width() / 2., y_pos,
            f'{height:.1f}%',
            ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
