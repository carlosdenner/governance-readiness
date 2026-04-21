import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Preprocessing System Origin using '22_dev_method' ---
# Previous attempt with '10_commercial_ai' failed as it contained use-case descriptions.
# '22_dev_method' contains 'Developed in-house.' and 'Developed with contracting resources.'

def classify_origin(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    if 'in-house' in val_str and 'contracting' not in val_str:
        return 'In-House'
    elif 'contracting' in val_str and 'in-house' not in val_str:
        return 'Commercial/Contractor'
    return np.nan # Exclude 'Both' or others for clear comparison

eo_df['system_origin'] = eo_df['22_dev_method'].apply(classify_origin)

# --- Preprocessing '38_code_access' ---
def check_access(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    # 'Yes - ...' counts as Yes.
    if val_str.startswith('yes'):
        return 1
    elif val_str.startswith('no'):
        return 0
    return np.nan

eo_df['has_code_access'] = eo_df['38_code_access'].apply(check_access)

# Filter analysis data
analysis_df = eo_df.dropna(subset=['system_origin', 'has_code_access'])

print(f"Analysis Data Shape: {analysis_df.shape}")
print("Group Counts:")
print(analysis_df['system_origin'].value_counts())

# --- Statistical Analysis ---
contingency_table = pd.crosstab(analysis_df['system_origin'], analysis_df['has_code_access'])
print("\nContingency Table (0=No Access, 1=Access):")
print(contingency_table)

if not contingency_table.empty:
    chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}\nP-value: {p:.4e}")
else:
    print("\nError: Contingency table is empty.")

# Calculate rates
rates = analysis_df.groupby('system_origin')['has_code_access'].mean()
print("\nCode Access Rates:")
print(rates)

# --- Visualization ---
plt.figure(figsize=(10, 6))
bars = plt.bar(rates.index, rates.values * 100, color=['#d62728', '#1f77b4'], alpha=0.8)

plt.title('Code Access Compliance: Commercial/Contractor vs. In-House AI', fontsize=14)
plt.ylabel('Percentage with Code Access (%)', fontsize=12)
plt.xlabel('System Origin', fontsize=12)
plt.ylim(0, 110)

# Add labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
