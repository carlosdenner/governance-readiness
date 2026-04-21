import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Columns
col_assess = '52_impact_assessment'
col_monitor = '56_monitor_postdeploy'

# Define Mapping Logic
def map_assessment(val):
    s = str(val).lower().strip()
    if s in ['yes', 'yes.']:
        return 1
    # 'Planned or in-progress', 'No', 'nan' -> 0
    return 0

def map_monitoring(val):
    s = str(val).lower().strip()
    if pd.isna(val) or s == 'nan':
        return 0
    
    # Positive indicators based on previous exploration
    if any(x in s for x in ['intermittent', 'automated', 'established process']):
        return 1
    
    # Negative indicators
    if any(x in s for x in ['no monitoring', 'not safety', 'under development']):
        return 0
        
    return 0

# Apply Mapping
eo_data['assess_bin'] = eo_data[col_assess].apply(map_assessment)
eo_data['monitor_bin'] = eo_data[col_monitor].apply(map_monitoring)

# Create Contingency Table
ct = pd.crosstab(eo_data['assess_bin'], eo_data['monitor_bin'])
ct = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

print("\nContingency Table (Assessment vs Monitoring):")
print(ct)
print("(Row=Assessment, Col=Monitoring; 0=No, 1=Yes)")

# Cells
no_assess_no_mon = ct.loc[0, 0]
no_assess_yes_mon = ct.loc[0, 1]
yes_assess_no_mon = ct.loc[1, 0]
yes_assess_yes_mon = ct.loc[1, 1]

# McNemar's Test
result = mcnemar(ct, exact=False, correction=True)

print(f"\n--- McNemar's Test Results ---")
print(f"Statistic (chi-squared): {result.statistic:.4f}")
print(f"P-value: {result.pvalue:.4e}")

# Analysis
total = len(eo_data)
print(f"\n--- Detailed Analysis ---")
print(f"Total Systems: {total}")
print(f"Assessment Completed: {eo_data['assess_bin'].sum()} ({(eo_data['assess_bin'].sum()/total)*100:.1f}%)")
print(f"Monitoring Established: {eo_data['monitor_bin'].sum()} ({(eo_data['monitor_bin'].sum()/total)*100:.1f}%)")

print(f"\nDiscordant Pairs:")
print(f"Paperwork Only (Assess=Yes, Mon=No): {yes_assess_no_mon}")
print(f"Ops Only (Assess=No, Mon=Yes): {no_assess_yes_mon}")

if yes_assess_no_mon > no_assess_yes_mon:
    ratio = yes_assess_no_mon / (no_assess_yes_mon if no_assess_yes_mon > 0 else 1)
    print(f"Result: The 'Paperwork-Operations' Gap is confirmed. Agencies are {ratio:.2f}x more likely to have Assessment without Monitoring than vice versa.")
else:
    print("Result: No significant Paperwork-Operations Gap detected in the expected direction.")

# Visualization
labels = ['Assess & Mon', 'Assess Only', 'Mon Only', 'Neither']
counts = [yes_assess_yes_mon, yes_assess_no_mon, no_assess_yes_mon, no_assess_no_mon]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, counts, color=['#2ca02c', '#ff7f0e', '#1f77b4', '#7f7f7f'])
plt.title('The Paperwork-Operations Gap (EO 13960)')
plt.ylabel('Number of AI Systems')
plt.bar_label(bars)
plt.show()
