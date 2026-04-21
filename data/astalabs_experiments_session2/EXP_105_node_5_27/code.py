import pandas as pd
import numpy as np
import scipy.stats as stats
import sys

# [debug] Print python version and current working directory to ensure environment is sane
# import os
# print(sys.version)
# print(os.getcwd())

print("Starting 'Shadow AI' Hypothesis Test...\n")

# 1. Load the dataset
try:
    # Try loading from the parent directory as instructed
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    print("Dataset loaded successfully from parent directory.")
except FileNotFoundError:
    # Fallback if running in same directory
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        print("Dataset loaded successfully from current directory.")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# 2. Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {df_eo.shape}")

# 3. Categorize Development Stage
# Inspecting unique values to ensure robust mapping (printing top 10 for verification)
raw_stages = df_eo['16_dev_stage'].astype(str).value_counts().head(10)
print("\nTop 10 Raw '16_dev_stage' values:")
print(raw_stages)

def map_stage(val):
    s = str(val).lower()
    # Operational keywords
    if any(x in s for x in ['oper', 'prod', 'use', 'maint', 'deploy', 'sustain']):
        return 'Operational'
    # Development keywords
    if any(x in s for x in ['dev', 'plan', 'acq', 'pilot', 'test', 'research', 'concept']):
        return 'Development'
    return 'Other'

df_eo['stage_group'] = df_eo['16_dev_stage'].apply(map_stage)

# Filter for analysis groups
df_analysis = df_eo[df_eo['stage_group'].isin(['Operational', 'Development'])].copy()

print("\nStage Group Distribution:")
print(df_analysis['stage_group'].value_counts())

# 4. Categorize ATO Status (Compliance)
# Goal: Identify 'Shadow AI' (Operational but No ATO)
# 'Yes' = Compliant, Anything else = Non-Compliant

def map_ato(val):
    if pd.isna(val):
        return 0 # Missing is Non-Compliant
    s = str(val).strip().lower()
    if s.startswith('yes'):
        return 1 # Compliant
    return 0 # Non-Compliant

df_analysis['has_ato'] = df_analysis['40_has_ato'].apply(map_ato)

# 5. Generate Contingency Table
# Rows: Stage, Columns: ATO Status (0=No, 1=Yes)
ct = pd.crosstab(df_analysis['stage_group'], df_analysis['has_ato'])
ct.columns = ['No ATO (Non-Compliant)', 'Has ATO (Compliant)']
print("\nContingency Table (Counts):")
print(ct)

# 6. Calculate Statistics
# Non-Compliance Rate by Stage
summary = df_analysis.groupby('stage_group')['has_ato'].agg(['count', 'sum'])
summary['non_compliant_count'] = summary['count'] - summary['sum']
summary['non_compliance_rate'] = summary['non_compliant_count'] / summary['count']

print("\nCompliance Analysis:")
print(summary[['count', 'non_compliant_count', 'non_compliance_rate']])

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-Square Test of Independence:\nStatistic: {chi2:.4f}, p-value: {p:.4e}")

# Risk Ratio Calculation
# Risk = Probability of Non-Compliance (No ATO)
# RR = Risk(Operational) / Risk(Development)

risk_op = summary.loc['Operational', 'non_compliance_rate']
risk_dev = summary.loc['Development', 'non_compliance_rate']
risk_ratio = risk_op / risk_dev if risk_dev > 0 else np.nan

print(f"\nRisk of Non-Compliance (Operational): {risk_op:.2%}")
print(f"Risk of Non-Compliance (Development): {risk_dev:.2%}")
print(f"Risk Ratio (Op / Dev): {risk_ratio:.4f}")

# Interpretation
print("\n--- Interpretation ---")
if risk_op > 0.10:
    print(f"EVIDENCE OF SHADOW AI: {risk_op:.1%} of Operational systems lack a valid ATO.")
else:
    print(f"Minimal Shadow AI: Only {risk_op:.1%} of Operational systems lack a valid ATO.")

if p < 0.05:
    print("The difference in compliance rates between stages is statistically significant.")
    if risk_ratio < 1:
        print("Operational systems are significantly MORE compliant than Development systems (Expected).")
    else:
        print("Operational systems are significantly LESS compliant than Development systems (Unexpected/Alarming).")
else:
    print("No statistically significant difference in compliance rates found between stages.")
