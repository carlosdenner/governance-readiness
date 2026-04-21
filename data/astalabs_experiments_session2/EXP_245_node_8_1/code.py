import pandas as pd
import scipy.stats as stats
import sys
import os

# Load dataset
paths = ['../astalabs_discovery_all_data.csv', 'astalabs_discovery_all_data.csv']
df = None
for p in paths:
    if os.path.exists(p):
        df = pd.read_csv(p, low_memory=False)
        break

if df is None:
    print("Dataset not found.")
    sys.exit(1)

# Filter for EO 13960 Scored table
subset = df[df['source_table'] == 'eo13960_scored'].copy()

# define columns
col_dev_method = '22_dev_method'
col_code_access = '38_code_access'

# Drop rows with NaN in critical columns
subset = subset.dropna(subset=[col_dev_method, col_code_access])

# Define Groups based on Development Method
# Hypothesis: Commercial (Contracted) vs Government (In-house)
def classify_source(val):
    val = str(val).strip()
    if 'contracting resources' in val and 'in-house' not in val:
        return 'Commercial'
    elif 'in-house' in val and 'contracting' not in val:
        return 'Government'
    else:
        return None

subset['group'] = subset[col_dev_method].apply(classify_source)

# Filter only for the two groups of interest
subset = subset[subset['group'].notna()]

# Define Code Access (Yes/No)
def classify_access(val):
    val = str(val).strip().upper()
    if val.startswith('YES'):
        return 'Yes'
    elif val.startswith('NO'):
        return 'No'
    else:
        return None

subset['access_binary'] = subset[col_code_access].apply(classify_access)
subset = subset[subset['access_binary'].notna()]

# Summary stats
group_counts = subset.groupby(['group', 'access_binary']).size().unstack(fill_value=0)
print("Contingency Table (Group vs Code Access):")
print(group_counts)

# Calculate percentages
comm_stats = group_counts.loc['Commercial']
gov_stats = group_counts.loc['Government']

comm_total = comm_stats.sum()
comm_yes = comm_stats.get('Yes', 0)
comm_rate = (comm_yes / comm_total) * 100 if comm_total > 0 else 0

gov_total = gov_stats.sum()
gov_yes = gov_stats.get('Yes', 0)
gov_rate = (gov_yes / gov_total) * 100 if gov_total > 0 else 0

print(f"\nCommercial (Contracted) Code Access Rate: {comm_rate:.1f}% ({comm_yes}/{comm_total})")
print(f"Government (In-house) Code Access Rate:   {gov_rate:.1f}% ({gov_yes}/{gov_total})")

# Statistical Test
if comm_total > 0 and gov_total > 0:
    contingency = group_counts.values
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Statistic: {chi2:.4f}")
    print(f"P-value: {p:.6e}")
    
    if p < 0.05:
        print("Result: Significant difference found.")
        if comm_rate < gov_rate:
            print("Supports Hypothesis: Commercial systems have significantly lower code access.")
        else:
            print("Contradicts Hypothesis: Commercial systems have higher code access.")
    else:
        print("Result: No significant difference found.")
else:
    print("Insufficient data for statistical test.")