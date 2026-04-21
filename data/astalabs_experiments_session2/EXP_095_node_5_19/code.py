import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"EO 13960 Scored Dataset Shape: {eo_data.shape}")

# --- Feature Engineering: Agency Type ---
# Inspect unique agencies to ensure mapping logic is sound
# print("Unique Agencies:", eo_data['3_agency'].unique()[:10])

security_keywords = ['Defense', 'Homeland Security', 'Justice', 'State']

def categorize_agency(agency_name):
    if pd.isna(agency_name):
        return 'Civilian' # Default to civilian if unknown, though rare
    agency_str = str(agency_name)
    for keyword in security_keywords:
        if keyword in agency_str:
            return 'Security'
    return 'Civilian'

eo_data['Agency_Type'] = eo_data['3_agency'].apply(categorize_agency)

# --- Feature Engineering: Code Access ---
target_col = '38_code_access'

# Print unique values to determine mapping logic
print(f"\nUnique values in {target_col}:\n", eo_data[target_col].value_counts(dropna=False))

# Mapping logic based on standard EO13960 responses
# Usually: "Yes", "No", "Yes, specific...", etc.
def categorize_code_access(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    # Positive indicators
    if val_str.startswith('yes') or 'open source' in val_str or 'available' in val_str or 'public' in val_str:
        return 1
    return 0

eo_data['Code_Access_Binary'] = eo_data[target_col].apply(categorize_code_access)

# --- Statistical Analysis ---
contingency_table = pd.crosstab(eo_data['Agency_Type'], eo_data['Code_Access_Binary'])
contingency_table.columns = ['No Access', 'Access Provided']

print("\n--- Contingency Table ---")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Odds Ratio calculation
# OR = (Security_No * Civilian_Yes) / (Security_Yes * Civilian_No) ?? 
# Let's use the odds of *Access* for Security vs Civilian
# Odds(Security) = Access / No Access
# Odds(Civilian) = Access / No Access
# Ratio = Odds(Security) / Odds(Civilian)

if 'Access Provided' in contingency_table.columns and 'No Access' in contingency_table.columns:
    sec_access = contingency_table.loc['Security', 'Access Provided']
    sec_no = contingency_table.loc['Security', 'No Access']
    civ_access = contingency_table.loc['Civilian', 'Access Provided']
    civ_no = contingency_table.loc['Civilian', 'No Access']

    try:
        odds_security = sec_access / sec_no
        odds_civilian = civ_access / civ_no
        odds_ratio = odds_security / odds_civilian
        print(f"\nOdds of Access (Security): {odds_security:.4f}")
        print(f"Odds of Access (Civilian): {odds_civilian:.4f}")
        print(f"Odds Ratio (Security/Civilian): {odds_ratio:.4f}")
    except ZeroDivisionError:
        print("\nCannot calculate Odds Ratio due to zero division.")

# --- Visualization ---
# Calculate percentages for plotting
plot_data = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

ax = plot_data.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#d9534f', '#5cb85c'])
plt.title('Code Access Transparency by Agency Mission Type')
plt.ylabel('Percentage of Use Cases')
plt.xlabel('Agency Type')
plt.legend(title='Code Access', loc='upper right')
plt.xticks(rotation=0)

# Annotate bars
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', weight='bold')

plt.tight_layout()
plt.show()