import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

# Load dataset
file_path = "astalabs_discovery_all_data.csv"
if not os.path.exists(file_path):
    file_path = "../astalabs_discovery_all_data.csv"

df = pd.read_csv(file_path, low_memory=False)

# Filter for relevant source table
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset size: {len(subset)}")

# --- Logic for 'is_public' ---
# Column 26: Public Service (Descriptive text = Yes, 'No'/Empty = No)
def map_public_service(val):
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if len(s) < 2:  # Filters out empty strings, single chars like ' '
        return 0
    if s.lower() in ['no', 'none', 'n/a']:
        return 0
    return 1  # Contains description of service

# Column 27: Public Info (Explicit 'Yes')
def map_public_info(val):
    if pd.isna(val):
        return 0
    if str(val).strip().lower() == 'yes':
        return 1
    return 0

subset['public_service_flag'] = subset['26_public_service'].apply(map_public_service)
subset['public_info_flag'] = subset['27_public_info'].apply(map_public_info)
subset['is_public'] = ((subset['public_service_flag'] == 1) | (subset['public_info_flag'] == 1)).astype(int)

# --- Logic for 'has_notice' ---
# Column 59: AI Notice
# Positives: 'Online', 'In-person', 'Email', 'Other', 'Telephone'
# Negatives: NaN, 'None of the above', 'N/A', 'Waived', 'Not safety'
def map_notice(val):
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    # Negative keywords
    negatives = ['none of the above', 'n/a', 'waived', 'not safety', 'nan']
    if any(neg in s for neg in negatives):
        return 0
    # If it's not negative and has content, assume positive notice info
    if len(s) > 2:
        return 1
    return 0

subset['has_notice'] = subset['59_ai_notice'].apply(map_notice)

# --- Analysis ---
print("\n--- Value Counts ---")
print(f"Public Systems (is_public=1): {subset['is_public'].sum()}")
print(f"Internal Systems (is_public=0): {len(subset) - subset['is_public'].sum()}")
print(f"Systems with Notice (has_notice=1): {subset['has_notice'].sum()}")

# Generate Contingency Table with reindexing to ensure 2x2
contingency_table = pd.crosstab(subset['is_public'], subset['has_notice'])
# Reindex to ensure all categories exist (0 and 1)
contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

contingency_table.index = ['Internal', 'Public']
contingency_table.columns = ['No Notice', 'Has Notice']

print("\n--- Contingency Table ---")
print(contingency_table)

# Percentages
internal_total = contingency_table.loc['Internal'].sum()
public_total = contingency_table.loc['Public'].sum()

internal_compliance = (contingency_table.loc['Internal', 'Has Notice'] / internal_total * 100) if internal_total > 0 else 0
public_compliance = (contingency_table.loc['Public', 'Has Notice'] / public_total * 100) if public_total > 0 else 0

print(f"\nInternal Compliance: {internal_compliance:.2f}% (n={internal_total})")
print(f"Public Compliance:   {public_compliance:.2f}% (n={public_total})")

# Statistical Test
if internal_total > 0 and public_total > 0:
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi2: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Significant difference found.")
        if public_compliance > internal_compliance:
            print("Hypothesis Supported: Public systems are more likely to have notice.")
        else:
            print("Hypothesis Refuted: Public systems are LESS likely to have notice.")
    else:
        print("Result: No significant difference.")
else:
    print("Insufficient data for test.")
