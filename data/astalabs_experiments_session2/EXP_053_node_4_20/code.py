import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import os

# Define file path
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit(1)

# Filter for EO 13960 Scored dataset
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored rows loaded: {len(eo_data)}")

# --- Map Development Stage ---
def robust_map_dev_stage(val):
    if pd.isna(val):
        return None
    
    val_str = str(val).lower().strip()
    
    # Operational keywords
    op_keywords = [
        'implemented', 'implementation', 'operation', 'operational', 
        'retired', 'production', 'mission', 'use', 'deployed'
    ]
    if any(k in val_str for k in op_keywords):
        return 'Operational'
    
    # Pre-Operational keywords
    pre_keywords = [
        'development', 'planning', 'planned', 'pilot', 
        'research', 'design', 'initiated', 'acquisition'
    ]
    if any(k in val_str for k in pre_keywords):
        return 'Pre-Operational'
        
    return 'Other'

eo_data['stage_group'] = eo_data['16_dev_stage'].apply(robust_map_dev_stage)

# Filter for relevant groups
analysis_df = eo_data[eo_data['stage_group'].isin(['Operational', 'Pre-Operational'])].copy()
print(f"Rows retained for analysis: {len(analysis_df)}")

# --- Map AI Notice (Corrected Logic) ---
def map_notice_corrected(val):
    if pd.isna(val):
        return 0
    
    val_str = str(val).strip().lower()
    
    # Negative indicators (No notice provided)
    negative_indicators = [
        'none of the above', 
        'n/a', 
        'waived',
        'not safety',
        'nan'
    ]
    
    if any(neg in val_str for neg in negative_indicators):
        return 0
    
    # If it's not negative, it implies some form of notice (Online, Email, In-person, Other)
    return 1

analysis_df['notice_binary'] = analysis_df['59_ai_notice'].apply(map_notice_corrected)

# --- Generate Contingency Table ---
contingency_table = pd.crosstab(analysis_df['stage_group'], analysis_df['notice_binary'])

# Ensure columns 0 and 1 exist
contingency_table = contingency_table.reindex(columns=[0, 1], fill_value=0)
contingency_table.columns = ['No Notice', 'Has Notice']

print("\nContingency Table:")
print(contingency_table)

# --- Statistical Analysis ---
total_positives = contingency_table['Has Notice'].sum()

if total_positives == 0:
    print("\n[Analysis Outcome] No positive 'AI Notice' cases found even with corrected mapping.")
else:
    # Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # Compliance Rates
    compliance_rates = analysis_df.groupby('stage_group')['notice_binary'].mean()
    print("\nCompliance Rates (Proportion with AI Notice):")
    print(compliance_rates)

    # Odds Ratio
    try:
        op_yes = contingency_table.loc['Operational', 'Has Notice']
        op_no = contingency_table.loc['Operational', 'No Notice']
        pre_yes = contingency_table.loc['Pre-Operational', 'Has Notice']
        pre_no = contingency_table.loc['Pre-Operational', 'No Notice']
        
        if op_no == 0 or pre_no == 0:
            print("Warning: Zero count in denominator (No Notice). Odds Ratio undefined.")
        elif pre_yes == 0:
             print("Warning: Zero count in Pre-Operational Yes. Odds Ratio undefined.")
        else:
            odds_op = op_yes / op_no
            odds_pre = pre_yes / pre_no
            odds_ratio = odds_op / odds_pre
            print(f"Odds Ratio (Operational vs Pre-Operational): {odds_ratio:.4f}")
    except Exception as e:
        print(f"Could not calculate Odds Ratio: {e}")

    # --- Visualization ---
    plt.figure(figsize=(8, 6))
    stages = ['Operational', 'Pre-Operational']
    rates = [compliance_rates.get(s, 0) for s in stages]

    bars = plt.bar(stages, rates, color=['#4CAF50','#FFC107'])
    plt.title('AI Notice Compliance by Development Stage')
    plt.ylabel('Compliance Rate')
    plt.xlabel('Development Stage')
    plt.ylim(0, max(rates) * 1.2 if max(rates) > 0 else 0.1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()