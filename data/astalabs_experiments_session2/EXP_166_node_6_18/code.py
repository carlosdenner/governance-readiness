import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt

# Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Filter for EO 13960 Scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# --- 1. Agency Categorization ---
def categorize_agency(row):
    agency = str(row['3_agency']).upper()
    abr = str(row['3_abr']).upper()
    
    # Defense / Intel / Security
    # Target: DOD, DOJ, DHS, State
    # Known abbreviations/substrings from debug: DHS, STATE, HOMELAND SECURITY, DEPARTMENT OF STATE
    defense_ids = ['DHS', 'DOD', 'DOJ', 'DOS', 'STATE', 'CIA', 'NSA', 'ODNI']
    defense_names = ['DEFENSE', 'JUSTICE', 'HOMELAND', 'STATE', 'INTELLIGENCE', 'SECURITY']
    
    if any(k == abr for k in defense_ids) or any(k in agency for k in defense_names):
         return 'Defense/Intel'

    # Civilian / Service
    # Target: HHS, VA, Education, DOE, DOT, HUD
    # Known abbreviations/substrings from debug: HHS, VA, DOE, DOT, HUD, ED, USDA, SSA
    civilian_ids = ['HHS', 'VA', 'DOT', 'DOE', 'HUD', 'ED', 'USDA', 'SSA', 'DOC', 'DOL', 'TREAS']
    civilian_names = ['HEALTH', 'VETERANS', 'TRANSPORTATION', 'ENERGY', 'HOUSING', 'EDUCATION', 'AGRICULTURE', 'SOCIAL SECURITY', 'COMMERCE', 'LABOR', 'TREASURY']

    if any(k == abr for k in civilian_ids) or any(k in agency for k in civilian_names):
        return 'Civilian/Service'
        
    return 'Other'

df_eo['Agency_Category'] = df_eo.apply(categorize_agency, axis=1)

# --- 2. Notice Compliance Cleaning ---
# Based on debug output:
# Compliant: 'Online', 'In-person', 'Email', 'Telephone', 'Other'
# Non-Compliant: 'None of the above', 'waived'
# Exclude: 'NaN', 'N/A - individuals are not interacting...', 'AI is not safety...'

def clean_notice(val):
    s = str(val).lower().strip()
    
    # Check for exclusions first
    if s == 'nan' or 'n/a' in s or 'not safety' in s:
        return np.nan
        
    # Check for Non-Compliance
    if 'none of the above' in s or 'waived' in s:
        return 0
        
    # Check for Compliance (Positive indicators)
    if any(x in s for x in ['online', 'in-person', 'email', 'telephone', 'other', 'terms']): 
        return 1
        
    return np.nan

df_eo['Notice_Compliance'] = df_eo['59_ai_notice'].apply(clean_notice)

# --- 3. Analysis ---

# Filter for valid rows (Known Agency Category AND Known Notice Status)
df_analysis = df_eo[
    (df_eo['Agency_Category'].isin(['Defense/Intel', 'Civilian/Service'])) &
    (df_eo['Notice_Compliance'].notna())
].copy()

print(f"Valid analysis rows: {len(df_analysis)}")

# Generate Stats
group_stats = df_analysis.groupby('Agency_Category')['Notice_Compliance'].agg(['count', 'mean', 'sum'])
group_stats['compliance_pct'] = group_stats['mean'] * 100

print("\n--- Compliance Statistics ---")
print(group_stats)

# Contingency Table
contingency_table = pd.crosstab(df_analysis['Agency_Category'], df_analysis['Notice_Compliance'])
print("\n--- Contingency Table (0=No Notice, 1=Notice) ---")
print(contingency_table)

# Check if we have data
if len(df_analysis) == 0:
    print("No valid data found for analysis after filtering.")
    exit(0)

# Statistical Test
# Use Fisher's Exact if sample size is small, otherwise Chi-square
if contingency_table.size == 4 and contingency_table.min().min() < 5:
    print("\nUsing Fisher's Exact Test (small sample size)...")
    odds_ratio, p_value = fisher_exact(contingency_table)
    test_name = "Fisher's Exact"
else:
    print("\nUsing Chi-Square Test...")
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    test_name = "Chi-Square"

print(f"\n--- {test_name} Results ---")
print(f"p-value: {p_value:.4e}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Result: Statistically Significant Difference found.")
else:
    print("Result: No Statistically Significant Difference.")

# Plot
plt.figure(figsize=(8, 5))
colors = ['#1f77b4', '#ff7f0e']
ax = group_stats['mean'].plot(kind='bar', color=colors, alpha=0.8, yerr=1.96 * np.sqrt(group_stats['mean']*(1-group_stats['mean'])/group_stats['count']), capsize=5)
plt.title('AI Public Notice Compliance: Defense vs Civilian')
plt.ylabel('Compliance Rate')
plt.xlabel('Agency Category')
plt.xticks(rotation=0)
plt.ylim(0, 1.1)

# Add labels
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()