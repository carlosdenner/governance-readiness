import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

print("Starting Malice by Sector analysis (Attempt 5)...")

# 1. Load Dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# 2. Filter for AIID Incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded AIID Incidents: {len(aiid)} rows")

# 3. Identify Columns
cols = aiid.columns.tolist()
intent_col = next((c for c in cols if 'Intentional Harm' in c), '82_Intentional Harm')
sector_col = next((c for c in cols if 'Sector of Deployment' in c), '78_Sector of Deployment')

# 4. Data Cleaning & Mapping
# Intentionality: Check if string starts with 'yes' (case insensitive)
aiid['is_intentional'] = aiid[intent_col].astype(str).str.lower().str.strip().str.startswith('yes')

# Sector Mapping
def classify_sector(val):
    v = str(val).lower()
    if any(k in v for k in ['defense', 'government', 'military', 'security', 'police', 'public safety', 'law enforcement', 'justice', 'surveillance']):
        return 'Security/Gov'
    elif any(k in v for k in ['health', 'medical', 'hospital', 'transport', 'vehicle', 'automotive', 'aviation', 'rail', 'flight', 'driverless']):
        return 'Safety-Critical/Civilian'
    else:
        return 'Other'

aiid['sector_group'] = aiid[sector_col].apply(classify_sector)

# 5. Filter for Analysis Groups
analysis_df = aiid[aiid['sector_group'] != 'Other'].copy()
print(f"\nRows retained for analysis: {len(analysis_df)}")
print("Group Counts:")
print(analysis_df['sector_group'].value_counts())
print("Intentionality Counts:")
print(analysis_df['is_intentional'].value_counts())

# 6. Create Robust Contingency Table
# Initialize with all possible keys to ensure 2x2 shape
contingency = pd.crosstab(analysis_df['sector_group'], analysis_df['is_intentional'])

# Explicitly reindex to ensure all rows/cols exist
expected_index = ['Safety-Critical/Civilian', 'Security/Gov']
expected_cols = [False, True]

contingency = contingency.reindex(index=expected_index, columns=expected_cols, fill_value=0)
contingency.columns = ['Unintentional', 'Intentional']

print("\n--- Contingency Table ---")
print(contingency)

# 7. Statistical Testing
# Fisher's Exact Test is suitable for 2x2 tables, especially with small counts
oddsratio, pvalue = stats.fisher_exact(contingency)
print(f"\nFisher's Exact Test Results:")
print(f"P-value: {pvalue:.4e}")
print(f"Odds Ratio: {oddsratio:.4f}")

# Calculate percentages for interpretation
probs = contingency.div(contingency.sum(axis=1), axis=0) * 100
print("\n--- Conditional Probabilities (%) ---")
print(probs)

# Compare Intentionality Rates
rate_gov = probs.loc['Security/Gov', 'Intentional']
rate_civ = probs.loc['Safety-Critical/Civilian', 'Intentional']
print(f"\nIntentionality Rate - Security/Gov: {rate_gov:.2f}%")
print(f"Intentionality Rate - Safety/Civilian: {rate_civ:.2f}%")

if pvalue < 0.05:
    print("CONCLUSION: The difference is statistically significant. Hypothesis Supported.")
else:
    print("CONCLUSION: The difference is NOT statistically significant. Hypothesis Rejected.")

# 8. Plotting
ax = contingency.plot(kind='bar', stacked=True, color=['#1f77b4', '#d62728'], rot=0)
plt.title('Harm Intentionality by Sector Group')
plt.xlabel('Sector Group')
plt.ylabel('Incident Count')
plt.legend(title='Intentionality')

# Annotate bars
for c in ax.containers:
    ax.bar_label(c, label_type='center', fmt='%d')

plt.tight_layout()
plt.show()