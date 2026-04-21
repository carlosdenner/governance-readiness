import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

col_autonomy = '57_autonomous_impact'
col_risks = '54_key_risks'

# --- Classification Logic ---

def classify_autonomy(val):
    s = str(val).lower().strip()
    if s == 'nan' or s == 'missing':
        return 'Unknown'
    
    # Autonomous: "Yes - All individual..." or "Other - Immediate human intervention is not practicable..."
    # Both imply the system acts without direct human control/intervention in the loop.
    if s.startswith('yes - all individual') or 'immediate human intervention is not practicable' in s:
        return 'Autonomous'
    
    # Human-Assisted: "No - Some individual decisions..."
    if s.startswith('no - some individual'):
        return 'Human-Assisted'
    
    return 'Unknown'

def classify_risk(val):
    # Treat NaN as 0 (No risk identified/documented)
    if pd.isna(val) or val == 'nan':
        return 0
    
    s = str(val).lower().strip()
    negative_terms = ['no', 'none', 'n/a', 'not applicable', 'none identified', '0', 'missing']
    
    # exact match check
    if s in negative_terms:
        return 0
    
    # distinct phrases check
    if 'no key risks identified' in s:
        return 0
    if s.startswith('n/a'):
        return 0
        
    # specific check for 'none.'
    if s == 'none.':
        return 0

    # If text exists and isn't a negative, assume risks are described/identified
    return 1

# Apply classification
eo_df['Autonomy_Class'] = eo_df[col_autonomy].apply(classify_autonomy)
eo_df['Risk_Flag'] = eo_df[col_risks].apply(classify_risk)

# Filter for analysis groups
analysis_df = eo_df[eo_df['Autonomy_Class'].isin(['Autonomous', 'Human-Assisted'])]

# --- Generate Statistics ---
print("--- Analysis of Automation-Overconfidence Paradox ---")

# Group counts
group_counts = analysis_df['Autonomy_Class'].value_counts()
print(f"\nSample Sizes:\n{group_counts}")

# Risk Identification Rates
risk_stats = analysis_df.groupby('Autonomy_Class')['Risk_Flag'].agg(['count', 'sum', 'mean'])
risk_stats.columns = ['Total', 'Risks_Identified', 'Rate']
print("\nRisk Identification Stats:")
print(risk_stats)

# Construct Contingency Table for Fisher's Exact Test
# Rows: Autonomous, Human-Assisted
# Cols: Risk Identified (1), Risk Not Identified (0)

auto_identified = risk_stats.loc['Autonomous', 'Risks_Identified']
auto_total = risk_stats.loc['Autonomous', 'Total']
auto_not = auto_total - auto_identified

human_identified = risk_stats.loc['Human-Assisted', 'Risks_Identified']
human_total = risk_stats.loc['Human-Assisted', 'Total']
human_not = human_total - human_identified

contingency_table = [[auto_identified, auto_not], [human_identified, human_not]]

print("\nContingency Table (Rows: Auto, Human; Cols: Identified, Not Identified):")
print(contingency_table)

# Fisher's Exact Test (Two-sided)
odds_ratio, p_value = fisher_exact(contingency_table, alternative='two-sided')

print(f"\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4e}")

alpha = 0.05
if p_value < alpha:
    print("Conclusion: Statistically significant difference found.")
else:
    print("Conclusion: No statistically significant difference found.")

# Interpretation helper
if risk_stats.loc['Autonomous', 'Rate'] < risk_stats.loc['Human-Assisted', 'Rate']:
    print("Direction: Autonomous systems have a LOWER risk identification rate.")
else:
    print("Direction: Autonomous systems have a HIGHER (or equal) risk identification rate.")
