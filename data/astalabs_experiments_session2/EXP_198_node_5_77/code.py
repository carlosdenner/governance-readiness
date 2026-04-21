import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# [debug] Start of experiment execution
print("Starting experiment: Intentionality of Failure Modes")

# 1. Load Data
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    # Fallback if file is in current directory
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    print("Dataset loaded from current directory.")

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)} rows")

# 2. Clean 'Intentional Harm' (Column index 82 usually, name 'Intentional Harm' or similar)
# Let's find the exact column name
intent_col = [c for c in aiid_df.columns if 'Intentional Harm' in c]
if not intent_col:
    # Fallback based on known schema from previous steps
    # The column might be named '82_Intentional Harm' or just 'Intentional Harm' depending on header processing
    # In the provided previous output, it was '82: Intentional Harm'. 
    # However, pandas usually reads the header. Let's look for partial match.
    intent_col = [c for c in aiid_df.columns if 'Intentional' in c]

if intent_col:
    intent_col = intent_col[0]
    print(f"Using column for Intent: '{intent_col}'")
else:
    print("Could not find Intentional Harm column. Listing columns:")
    print(aiid_df.columns.tolist()[:20])
    exit(1)

# Normalize Intent
# Assuming values like 'Yes', 'No', 'True', 'False', or boolean
aiid_df['intent_clean'] = aiid_df[intent_col].astype(str).str.lower().map({
    'yes': 'Intentional',
    'true': 'Intentional',
    '1': 'Intentional',
    '1.0': 'Intentional',
    'no': 'Unintentional',
    'false': 'Unintentional',
    '0': 'Unintentional',
    '0.0': 'Unintentional'
}).fillna('Unintentional') # Treat NaNs as Unintentional for now, or exclude. 
# Let's verify distribution
print("Intent distribution:")
print(aiid_df['intent_clean'].value_counts())

# 3. Categorize 'Known AI Technical Failure' 
# Find column
tech_col = [c for c in aiid_df.columns if 'Technical Failure' in c and 'Known' in c]
if tech_col:
    tech_col = tech_col[0]
    print(f"Using column for Technical Failure: '{tech_col}'")
else:
    print("Could not find Technical Failure column.")
    exit(1)

def categorize_failure(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).lower()
    
    # Security / Privacy keywords
    security_keywords = ['adversarial', 'attack', 'privacy', 'security', 'leakage', 
                         'extraction', 'inversion', 'poisoning', 'evasion', 'model theft']
    for kw in security_keywords:
        if kw in val_str:
            return 'Security/Privacy'
            
    # Safety / Reliability keywords
    safety_keywords = ['error', 'bias', 'fairness', 'robustness', 'safety', 'accident', 
                       'hallucination', 'malfunction', 'performance', 'reliability', 'unsafe']
    for kw in safety_keywords:
        if kw in val_str:
            return 'Safety/Reliability'
            
    return 'Other'

aiid_df['failure_category'] = aiid_df[tech_col].apply(categorize_failure)

# Filter out 'Unknown' and 'Other' to focus on the hypothesis comparison if needed, 
# but let's keep 'Other' to see the full picture, remove 'Unknown' for stats.
analysis_df = aiid_df[aiid_df['failure_category'] != 'Unknown'].copy()

print("\nFailure Category Distribution:")
print(analysis_df['failure_category'].value_counts())

# 4. Contingency Table
contingency = pd.crosstab(analysis_df['intent_clean'], analysis_df['failure_category'])
print("\nContingency Table:")
print(contingency)

# 5. Chi-square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# 6. Residual Analysis (Standardized Residuals)
# (Observed - Expected) / sqrt(Expected)
residuals = (contingency - expected) / np.sqrt(expected)
print("\nStandardized Residuals:")
print(residuals)

# Identify overrepresented pairs
print("\nInterpretation:")
if p < 0.05:
    print("Significant relationship found between Intent and Failure Mode.")
    # Check specific hypothesis cells
    try:
        sec_resid = residuals.loc['Intentional', 'Security/Privacy']
        safe_resid = residuals.loc['Unintentional', 'Safety/Reliability']
        print(f"Residual for Intentional -> Security/Privacy: {sec_resid:.2f}")
        print(f"Residual for Unintentional -> Safety/Reliability: {safe_resid:.2f}")
        
        if sec_resid > 1.96:
            print("Confirmed: Intentional incidents are significantly associated with Security/Privacy failures.")
        else:
            print("Intentional incidents are NOT significantly associated with Security/Privacy failures.")
            
        if safe_resid > 1.96:
            print("Confirmed: Unintentional incidents are significantly associated with Safety/Reliability failures.")
    except KeyError:
        print("Could not compute specific residuals due to missing categories in data.")
else:
    print("No significant relationship found.")

# Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Intent vs Technical Failure Mode')
plt.ylabel('Intent')
plt.xlabel('Failure Category')
plt.tight_layout()
plt.show()

# Bar chart of percentages
contingency_pct = pd.crosstab(analysis_df['intent_clean'], analysis_df['failure_category'], normalize='index') * 100
contingency_pct.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Technical Failure Modes by Intent')
plt.ylabel('Percentage')
plt.xlabel('Intent')
plt.legend(title='Failure Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()