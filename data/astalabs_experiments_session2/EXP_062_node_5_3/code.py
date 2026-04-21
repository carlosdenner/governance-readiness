import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# [debug]
print("Starting experiment: Agency Maturity Effect")

# 1. Load Data
file_path = '../astalabs_discovery_all_data.csv'
try:
    # Only loading necessary columns to save memory/time if possible, but sparse layout makes it tricky.
    # Loading all and filtering is safer given the structure.
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running in a different environment structure
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded EO 13960 data: {len(eo_data)} rows")

# 2. Agency Maturity Analysis
# Count use cases per agency
agency_counts = eo_data['3_agency'].value_counts().reset_index()
agency_counts.columns = ['3_agency', 'case_count']

# Determine quartiles
q1 = agency_counts['case_count'].quantile(0.25)
q3 = agency_counts['case_count'].quantile(0.75)
print(f"Quartiles -> Q1 (Nascent threshold): {q1}, Q3 (Mature threshold): {q3}")

# Label Agencies
def categorize_maturity(count):
    if count >= q3:
        return 'Mature'
    elif count <= q1:
        return 'Nascent'
    else:
        return 'Middle'

agency_counts['maturity'] = agency_counts['case_count'].apply(categorize_maturity)

# Merge maturity back to main data
eo_data = eo_data.merge(agency_counts[['3_agency', 'maturity', 'case_count']], on='3_agency', how='left')

# 3. Process Independent Evaluation Target
# Check values
target_col = '55_independent_eval'
unique_vals = eo_data[target_col].unique()
print(f"Unique values in {target_col}: {unique_vals}")

# Map to binary. Assuming variations of 'Yes'/'No'. NaN is treated as 0 (No) for governance scoring often, 
# but strict comparison might require dropping. Let's inspect and map cautiously.
# If it's a string, we normalize.
def clean_eval(val):
    if pd.isna(val):
        return 0 # Treat missing as lack of evidence/No in this context? Or np.nan?
                 # In government inventories, blank often means 'No'. Let's assume 0 but print warning if high NaN.
    s = str(val).lower().strip()
    if 'yes' in s or 'true' in s:
        return 1
    return 0

eo_data['has_eval'] = eo_data[target_col].apply(clean_eval)

# 4. Statistical Test (Mature vs Nascent)
analysis_set = eo_data[eo_data['maturity'].isin(['Mature', 'Nascent'])]

# Group stats
group_stats = analysis_set.groupby('maturity')['has_eval'].agg(['count', 'mean', 'sum'])
print("\n--- Group Statistics (Independent Evaluation Rate) ---")
print(group_stats)

# Contingency Table for Chi-Square
# Rows: [Mature, Nascent], Cols: [Eval=1, Eval=0]
mature_success = analysis_set[analysis_set['maturity'] == 'Mature']['has_eval'].sum()
mature_fail = analysis_set[analysis_set['maturity'] == 'Mature']['has_eval'].count() - mature_success

nascent_success = analysis_set[analysis_set['maturity'] == 'Nascent']['has_eval'].sum()
nascent_fail = analysis_set[analysis_set['maturity'] == 'Nascent']['has_eval'].count() - nascent_success

contingency_table = [[mature_success, mature_fail], [nascent_success, nascent_fail]]
chi2, p, dof, ex = stats.chi2_contingency(contingency_table)

print(f"\nChi-Square Test Results:")
print(f"Contingency Table (Rows: Mature, Nascent; Cols: Has_Eval, No_Eval):\n{contingency_table}")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# 5. Visualization: Evaluation Rate vs Log(Agency Case Count)
# We calculate rate per agency for the scatter plot
agency_perf = eo_data.groupby('3_agency').agg(
    case_count=('case_count', 'first'),
    eval_rate=('has_eval', 'mean'),
    maturity=('maturity', 'first')
).reset_index()

agency_perf['log_count'] = np.log1p(agency_perf['case_count'])

plt.figure(figsize=(10, 6))
colors = {'Mature': 'green', 'Nascent': 'red', 'Middle': 'gray'}

for mat, color in colors.items():
    subset = agency_perf[agency_perf['maturity'] == mat]
    plt.scatter(subset['log_count'], subset['eval_rate'], 
                label=mat, color=color, alpha=0.6, edgecolors='w', s=80)

# Trendline (using all data)
z = np.polyfit(agency_perf['log_count'], agency_perf['eval_rate'], 1)
p_poly = np.poly1d(z)
plt.plot(agency_perf['log_count'], p_poly(agency_perf['log_count']), "b--", alpha=0.5, label='Trendline')

plt.title('Agency Maturity vs Governance Rigor (Independent Eval)')
plt.xlabel('Log(Agency Case Count)')
plt.ylabel('Independent Evaluation Rate')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
