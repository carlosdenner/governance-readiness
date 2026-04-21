import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import sys
import os

# Define dataset path
dataset_filename = 'astalabs_discovery_all_data.csv'
path = f"../{dataset_filename}" if os.path.exists(f"../{dataset_filename}") else dataset_filename

print(f"Loading dataset from {path}...")
df = pd.read_csv(path, low_memory=False)

# Filter for AIID Incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid)} rows")

# --- COLUMN DISCOVERY ---
# Function to score columns based on keyword matches
def find_best_column(df, keywords):
    best_col = None
    max_matches = 0
    for col in df.columns:
        try:
            # Get unique values as string
            unique_vals = df[col].dropna().astype(str).str.lower().unique()
            match_count = sum(1 for val in unique_vals if any(k in val for k in keywords))
            if match_count > max_matches:
                max_matches = match_count
                best_col = col
        except:
            continue
    return best_col

# Keywords for Harm (looking for 'Civil Rights', 'Physical Safety', etc.)
harm_keywords = ['civil rights', 'physical safety', 'discrimination', 'privacy', 'injury', 'death', 'property damage']
# Keywords for Failure (looking for 'Robustness', 'Specification', 'Generalization')
fail_keywords = ['robustness', 'specification', 'generalization', 'adversarial', 'objective', 'distributional']

harm_col = find_best_column(aiid, harm_keywords)
fail_col = find_best_column(aiid, fail_keywords)

print(f"Identified Harm Column: '{harm_col}'")
print(f"Identified Failure Column: '{fail_col}'")

if not harm_col or not fail_col:
    print("Could not identify necessary columns. Printing 'harm' columns for inspection:")
    harm_cols_names = [c for c in aiid.columns if 'harm' in c.lower()]
    for c in harm_cols_names:
        print(f"Column: {c}")
        print(aiid[c].dropna().unique()[:5])
    sys.exit(1)

# --- CLEANING & MAPPING ---
aiid_clean = aiid.dropna(subset=[harm_col, fail_col]).copy()

def map_harm(val):
    val = str(val).lower()
    if any(x in val for x in ['physical', 'safety', 'death', 'injury', 'life', 'property', 'kill', 'accident']):
        return 'Physical/Safety'
    if any(x in val for x in ['civil rights', 'bias', 'discrimination', 'fairness', 'privacy', 'surveillance', 'policing', 'arrest', 'detention']):
        return 'Rights/Social'
    return 'Other'

def map_failure(val):
    val = str(val).lower()
    # Robustness: System fails under stress, shift, or attack
    # Note: 'Generalization Failure' is a robustness issue (fails on new distribution)
    if any(x in val for x in ['robustness', 'adversarial', 'generalization', 'distribution', 'shift', 'reliability', 'perturbation', 'noise', 'environmental', 'underfitting', 'overfitting']):
        return 'Robustness'
    # Specification: System aligns with wrong goal or has unintended side effects
    # Note: 'Underspecification' is a specification issue
    if any(x in val for x in ['specification', 'objective', 'reward', 'align', 'proxy', 'gaming', 'unintended', 'side effect', 'instruction']):
        return 'Specification'
    return 'Other'

aiid_clean['Harm_Group'] = aiid_clean[harm_col].apply(map_harm)
aiid_clean['Failure_Group'] = aiid_clean[fail_col].apply(map_failure)

# --- ANALYSIS ---
analysis_df = aiid_clean[
    (aiid_clean['Harm_Group'].isin(['Physical/Safety', 'Rights/Social'])) & 
    (aiid_clean['Failure_Group'].isin(['Robustness', 'Specification']))
]

print("\n--- Analysis Data Distribution ---")
print(analysis_df['Harm_Group'].value_counts())
print(analysis_df['Failure_Group'].value_counts())

if len(analysis_df) < 5:
    print("\nInsufficient data. Printing sample values from identified columns:")
    print(f"Harm Col ({harm_col}) samples:", aiid_clean[harm_col].unique()[:5])
    print(f"Fail Col ({fail_col}) samples:", aiid_clean[fail_col].unique()[:5])
else:
    contingency = pd.crosstab(analysis_df['Harm_Group'], analysis_df['Failure_Group'])
    print("\nContingency Table:\n", contingency)
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f'Harm vs Failure\n(Source: {harm_col} / {fail_col})')
    plt.ylabel('Harm Domain')
    plt.xlabel('Technical Failure')
    plt.show()

    # Interpretation
    print("\n--- Conclusion ---")
    if p < 0.05:
        print("Significant Association (Reject Null).")
        # Check directionality
        # Row percentages
        row_pcts = contingency.div(contingency.sum(axis=1), axis=0)
        print("\nRow Percentages:\n", row_pcts)
        
        # Check if Physical is mostly Robustness
        phys_robust = row_pcts.loc['Physical/Safety', 'Robustness'] if 'Physical/Safety' in row_pcts.index and 'Robustness' in row_pcts.columns else 0
        # Check if Rights is mostly Specification
        rights_spec = row_pcts.loc['Rights/Social', 'Specification'] if 'Rights/Social' in row_pcts.index and 'Specification' in row_pcts.columns else 0
        
        print(f"\nPhysical -> Robustness: {phys_robust:.1%}")
        print(f"Rights -> Specification: {rights_spec:.1%}")
        
        if phys_robust > 0.5 and rights_spec > 0.5:
            print("The data supports the hypothesis.")
        else:
            print("The data shows an association, but it may differ from the strict hypothesis.")
    else:
        print("No Significant Association (Fail to Reject Null).")
