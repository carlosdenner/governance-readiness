import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import sys

# [debug] Print standard start message
print("Starting Financial Malice Hypothesis Experiment...\n")

# 1. Load the dataset with robust path checking
file_name = 'astalabs_discovery_all_data.csv'
possible_paths = [file_name, f'../{file_name}']
data_path = None

for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    print(f"Error: Could not find {file_name} in current or parent directory.")
    # Debugging info to help locate file if this fails again
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    try:
        print("Files in parent directory:", os.listdir('..'))
    except Exception as e:
        print("Could not list parent directory:", e)
    sys.exit(1)

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path, low_memory=False)

# 2. Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents subset shape: {aiid_df.shape}")

# 3. Identify Column Names
# Based on metadata, columns of interest are 'Sector of Deployment' and 'Intentional Harm'
sector_col = 'Sector of Deployment'
intent_col = 'Intentional Harm'

# Verify columns exist
missing_cols = [c for c in [sector_col, intent_col] if c not in aiid_df.columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols}")
    print("Available columns:", aiid_df.columns.tolist())
    sys.exit(1)

# 4. Standardize Sector and Intentionality
aiid_df['sector_norm'] = aiid_df[sector_col].astype(str).str.lower()
aiid_df['intent_norm'] = aiid_df[intent_col].astype(str).str.lower()

# Define Filters
# Finance: 'financial', 'insurance', 'finance'
# Healthcare: 'healthcare', 'medical', 'health'
finance_mask = aiid_df['sector_norm'].str.contains('financ|insur', na=False)
health_mask = aiid_df['sector_norm'].str.contains('health|medic', na=False)

finance_df = aiid_df[finance_mask]
health_df = aiid_df[health_mask]

print(f"Finance/Insurance incidents found: {len(finance_df)}")
print(f"Healthcare incidents found: {len(health_df)}")

# Define Intentionality Logic
# We consider 'true' or 'yes' as intentional. Note: AIID data often has 'true'/'false' strings or booleans.
def is_intentional(val):
    v = str(val).lower()
    return 'true' in v or 'yes' in v

finance_intent_count = finance_df['intent_norm'].apply(is_intentional).sum()
health_intent_count = health_df['intent_norm'].apply(is_intentional).sum()

n_finance = len(finance_df)
n_health = len(health_df)

if n_finance == 0 or n_health == 0:
    print("Error: One of the sectors has 0 records. Cannot perform statistical test.")
    sys.exit(0)

prop_finance = finance_intent_count / n_finance
prop_health = health_intent_count / n_health

print(f"\n--- Results ---")
print(f"Finance: {finance_intent_count}/{n_finance} ({prop_finance:.2%}) intentional")
print(f"Healthcare: {health_intent_count}/{n_health} ({prop_health:.2%}) intentional")

# 5. Statistical Test (Two-proportion Z-test)
# Pooled proportion
p_pooled = (finance_intent_count + health_intent_count) / (n_finance + n_health)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_finance + 1/n_health))

if se == 0:
    print("Standard Error is 0, cannot calculate Z-score (proportions might be identical or zero).")
    z_score = 0
    p_value = 1.0
else:
    z_score = (prop_finance - prop_health) / se
    p_value = stats.norm.sf(abs(z_score)) * 2  # Two-tailed test

print(f"\nZ-score: {z_score:.4f}")
print(f"P-value: {p_value:.4e}")

if p_value < 0.05:
    print("Conclusion: Statistically Significant Difference.")
else:
    print("Conclusion: No Statistically Significant Difference.")

# 6. Visualization
labels = ['Finance & Insurance', 'Healthcare']
intent_rates = [prop_finance, prop_health]
accidental_rates = [1-prop_finance, 1-prop_health]

fig, ax = plt.subplots(figsize=(8, 6))

bar_width = 0.5
x_pos = np.arange(len(labels))

# Stacked bar chart
p1 = ax.bar(x_pos, intent_rates, bar_width, label='Intentional Harm', color='#d62728', alpha=0.8)
p2 = ax.bar(x_pos, accidental_rates, bar_width, bottom=intent_rates, label='Accidental/Other', color='#1f77b4', alpha=0.6)

ax.set_ylabel('Proportion of Incidents')
ax.set_title('Intentional Harm Rate: Finance vs Healthcare')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')

# Add percentage labels
for i, v in enumerate(intent_rates):
    if v > 0.05: # Only label if bar is visible enough
        ax.text(i, v/2, f"{v:.1%}", ha='center', va='center', color='white', fontweight='bold')

for i, v in enumerate(accidental_rates):
    if v > 0.05:
        ax.text(i, intent_rates[i] + v/2, f"{v:.1%}", ha='center', va='center', color='white')

plt.tight_layout()
plt.show()
