import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for the relevant source table
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# 1. Clean '52_impact_assessment'
def clean_assessment(val):
    s = str(val).lower().strip()
    if s in ['yes', 'true', '1']:
        return 'Yes'
    elif s in ['no', 'false', '0']:
        return 'No'
    return np.nan # Drop 'planned', 'nan', etc. for strict comparison

eo_data['Has_Assessment'] = eo_data['52_impact_assessment'].apply(clean_assessment)

# 2. Clean '62_disparity_mitigation' (Text Classification)
def classify_mitigation(val):
    if pd.isna(val):
        return 'No'
    
    text = str(val).lower().strip()
    
    # Strong negative indicators
    negative_keywords = [
        'n/a', 'none', 'not applicable', 'no demographic', 
        'not using pii', 'not safety', 'commercial solution',
        'no analysis', 'not tied to a demographic', 'does not use'
    ]
    
    # Strong positive indicators (override negatives if context implies action)
    positive_keywords = [
        'test', 'evaluat', 'monitor', 'review', 'audit', 'human',
        'bias', 'fairness', 'mitigat', 'check', 'assess', 
        'analysis', 'guardrail', 'feedback', 'retrain'
    ]
    
    # Logic: specific override for "N/A... but we do X"
    # If text is very short and contains negative, it's No.
    # If text contains positive keywords, it's likely Yes, even if it says "N/A for X, but Y"
    
    # Simple scoring for this experiment
    has_positive = any(k in text for k in positive_keywords)
    has_negative = any(k in text for k in negative_keywords)
    
    if has_positive:
        return 'Yes'
    elif has_negative:
        return 'No'
    else:
        # specific phrases from debug review
        if 'manual' in text or 'threshold' in text:
            return 'Yes'
        return 'No' # Default to No if ambiguous or empty

eo_data['Has_Mitigation'] = eo_data['62_disparity_mitigation'].apply(classify_mitigation)

# Filter for valid assessment rows
valid_df = eo_data.dropna(subset=['Has_Assessment'])

print(f"Data shape after filtering for valid Assessments: {valid_df.shape}")

# Generate Contingency Table
contingency_table = pd.crosstab(valid_df['Has_Assessment'], valid_df['Has_Mitigation'])
print("\nContingency Table (Rows: Assessment, Cols: Mitigation):")
print(contingency_table)

# Calculate Probabilities
if 'Yes' in contingency_table.index:
    ass_yes_total = contingency_table.loc['Yes'].sum()
    mit_yes_given_ass_yes = contingency_table.loc['Yes', 'Yes'] if 'Yes' in contingency_table.columns else 0
    prop_mit_given_ass = (mit_yes_given_ass_yes / ass_yes_total) * 100 if ass_yes_total > 0 else 0
else:
    prop_mit_given_ass = 0
    
if 'No' in contingency_table.index:
    ass_no_total = contingency_table.loc['No'].sum()
    mit_yes_given_ass_no = contingency_table.loc['No', 'Yes'] if 'Yes' in contingency_table.columns else 0
    prop_mit_given_no_ass = (mit_yes_given_ass_no / ass_no_total) * 100 if ass_no_total > 0 else 0
else:
    prop_mit_given_no_ass = 0

print(f"\n% of Systems WITH Assessment that have Mitigation: {prop_mit_given_ass:.2f}%")
print(f"% of Systems WITHOUT Assessment that have Mitigation: {prop_mit_given_no_ass:.2f}%")

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4e}")

# Plot
ct_norm = pd.crosstab(valid_df['Has_Assessment'], valid_df['Has_Mitigation'], normalize='index') * 100
if not ct_norm.empty:
    ax = ct_norm.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], figsize=(8, 6))
    plt.title('Impact of Assessment on Disparity Mitigation Controls')
    plt.ylabel('Percentage')
    plt.xlabel('Has Impact Assessment')
    plt.legend(title='Has Disparity Mitigation')
    plt.xticks(rotation=0)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
    plt.tight_layout()
    plt.show()
