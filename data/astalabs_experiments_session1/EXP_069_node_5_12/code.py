import json
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Handle file path: Try current directory first, as previous attempt with '../' failed
file_name = 'step3_enrichments.json'
if os.path.exists(file_name):
    file_path = file_name
elif os.path.exists(os.path.join('..', file_name)):
    file_path = os.path.join('..', file_name)
else:
    # Fallback to absolute check or raise error
    raise FileNotFoundError(f"{file_name} not found in current or parent directory.")

print(f"Loading dataset from: {file_path}")

with open(file_path, 'r') as f:
    data = json.load(f)

# Prepare data for analysis
records = []
for entry in data:
    # specific fields needed
    missing_controls = str(entry.get('missing_controls', '')).lower()
    harm_type = entry.get('harm_type', '').lower()
    
    # 1. Determine if Human-in-the-Loop is a missing control
    # Keywords: 'human', 'approval', 'override'
    # Note: Checking for 'human' might be too broad (e.g. 'humanoid'), but given the domain 'human-in-the-loop' or 'human oversight' is likely.
    # 'approval' covers 'approval gates', 'override' covers 'human override'.
    has_human_missing = any(keyword in missing_controls for keyword in ['human', 'approval', 'override'])
    
    # 2. Categorize Harm Type
    if harm_type in ['reliability', 'safety', 'physical_safety']:
        harm_category = 'Safety_Reliability'
    elif harm_type == 'security':
        harm_category = 'Security'
    else:
        harm_category = None # Exclude other harm types (privacy, bias, etc.)
    
    if harm_category:
        records.append({
            'case_study_id': entry.get('case_study_id'),
            'missing_controls': missing_controls,
            'harm_type': harm_type,
            'Missing_Human_Control': has_human_missing,
            'Harm_Category': harm_category
        })

# Create DataFrame
df = pd.DataFrame(records)

if df.empty:
    print("No records matched the criteria.")
else:
    print("=== Data Filtering Summary ===")
    print(f"Total records matching criteria: {len(df)}")
    print("Harm Category Counts:")
    print(df['Harm_Category'].value_counts())
    
    print("\n=== Missing Human Control Distribution (Counts) ===")
    # Group by Harm Category and Missing Human Control
    group_counts = df.groupby(['Harm_Category', 'Missing_Human_Control']).size().unstack(fill_value=0)
    print(group_counts)

    # Create Contingency Table for Stats
    # Rows: Harm Category, Cols: Missing Human Control
    contingency_table = pd.crosstab(df['Harm_Category'], df['Missing_Human_Control'])
    print("\n=== Contingency Table ===")
    print(contingency_table)

    # Ensure 2x2 table for Fisher's Exact Test
    # Fisher's test requires a 2x2 matrix.
    # If one category has 0 missing controls, the shape might be wrong or the unstack might miss columns.
    # We force a 2x2 structure
    ct_2x2 = pd.DataFrame(
        index=['Safety_Reliability', 'Security'], 
        columns=[True, False]
    ).fillna(0)
    
    # Update with actual values
    for idx in ct_2x2.index:
        for col in ct_2x2.columns:
            try:
                ct_2x2.loc[idx, col] = contingency_table.loc[idx, col]
            except KeyError:
                ct_2x2.loc[idx, col] = 0
    
    print("\n=== 2x2 Matrix for Fisher's Test ===")
    print(ct_2x2)

    # Perform Statistical Test
    # Hypothesis: Safety/Reliability is associated with 'True' (Missing Human Control) more than Security is.
    # We use Fisher's Exact Test.
    odds_ratio, p_value = stats.fisher_exact(ct_2x2)

    print("\n=== Fisher's Exact Test Results ===")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(ct_2x2.astype(int), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Missing "Human-in-the-Loop" Controls by Harm Type')
    plt.ylabel('Harm Category')
    plt.xlabel('Missing Human Control Identified?')
    plt.tight_layout()
    plt.show()
