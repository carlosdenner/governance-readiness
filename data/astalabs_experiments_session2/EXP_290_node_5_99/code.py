import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# [debug]
print("Starting experiment: Safety-Governance Gap (Attempt 2)")

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        print("Dataset loaded successfully (local).")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Filter for relevant source table
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Filtered for eo13960_scored: {len(df_eo)} rows")

# Normalize column names for easier access if needed (using existing names)
agency_col = '3_agency'
eval_col = '55_independent_eval'

# Define Groups based on keywords in agency name
def classify_agency(agency_name):
    if pd.isna(agency_name):
        return None
    agency_upper = str(agency_name).upper()
    
    safety_keywords = ['HEALTH', 'ENERGY', 'TRANSPORTATION', 'HOMELAND']
    admin_keywords = ['EDUCATION', 'COMMERCE', 'TREASURY']
    
    for kw in safety_keywords:
        if kw in agency_upper:
            return 'Safety-Critical'
    
    for kw in admin_keywords:
        if kw in agency_upper:
            return 'Administrative'
            
    return 'Other'

df_eo['group'] = df_eo[agency_col].apply(classify_agency)

# Filter only for the two groups of interest
df_analysis = df_eo[df_eo['group'].isin(['Safety-Critical', 'Administrative'])].copy()
print(f"Rows after group filtering: {len(df_analysis)}")

# Process the target variable '55_independent_eval'
# Updated logic: Check if string starts with 'Yes'
def parse_eval(val):
    if pd.isna(val):
        return 0
    # Normalize string
    s = str(val).strip().upper()
    if s.startswith('YES'):
        return 1
    return 0

df_analysis['has_eval'] = df_analysis[eval_col].apply(parse_eval)

# Verify parsing
print("\nValue counts for has_eval:")
print(df_analysis['has_eval'].value_counts())

# Contingency Table
contingency_table = pd.crosstab(df_analysis['group'], df_analysis['has_eval'])
print("\nContingency Table (Raw):")
print(contingency_table)

# Handle column naming safely based on what is present
# has_eval can be 0, 1, or both.
mapping = {0: 'No Eval', 1: 'Has Eval'}
contingency_table.columns = [mapping.get(c, c) for c in contingency_table.columns]

print("\nContingency Table (Labeled):")
print(contingency_table)

# Calculate proportions
props = df_analysis.groupby('group')['has_eval'].agg(['mean', 'count', 'sum'])
props.columns = ['Proportion', 'Total Count', 'Eval Count']
print("\nProportions:")
print(props)

# Statistical Test (Chi-Square)
# Only run if we have data in both groups and variance in the dependent variable
if contingency_table.shape == (2, 2):
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4f}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Statistically significant difference found.")
    else:
        print("Result: No statistically significant difference found.")
        
    # Visualization
    plt.figure(figsize=(8, 6))
    groups = props.index
    means = props['Proportion']
    
    # Plot
    bars = plt.bar(groups, means, color=['skyblue', 'salmon'])
    plt.title('Proportion of AI Systems with Independent Evaluations')
    plt.ylabel('Proportion')
    plt.ylim(0, max(means.max() * 1.2, 0.1)) # Scale y-axis
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.2%}',
                 ha='center', va='bottom')
                 
    plt.show()
else:
    print("Insufficient data structure for Chi-Square test (expected 2x2 table).")
    print(f"Current shape: {contingency_table.shape}")
