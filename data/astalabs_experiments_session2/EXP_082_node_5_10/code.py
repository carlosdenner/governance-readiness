import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Filter for EO13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 Data Shape: {eo_data.shape}")

# --- 1. Define Groups (Development Method) ---
# Specific strings found in debug step
IN_HOUSE_STR = "Developed in-house."
CONTRACTOR_STR = "Developed with contracting resources."

def classify_dev(val):
    if pd.isna(val):
        return None
    if val == IN_HOUSE_STR:
        return "In-house"
    if val == CONTRACTOR_STR:
        return "Contractor"
    return None

eo_data['dev_group'] = eo_data['22_dev_method'].apply(classify_dev)

# Filter only for the two groups of interest
analysis_df = eo_data[eo_data['dev_group'].notna()].copy()

# --- 2. Define Outcome (Continuous Monitoring) ---
# Mapping based on debugged values
# Positive indicators: 'Intermittent', 'Automated', 'Established'
# Negative/Null indicators: 'No monitoring', 'AI is not safety', NaN

def classify_monitor(val):
    if pd.isna(val):
        return 0 # Treating missing as 'No Monitoring'
    val_str = str(val)
    if "Intermittent" in val_str or "Automated" in val_str or "Established" in val_str:
        return 1
    return 0

analysis_df['is_monitored'] = analysis_df['56_monitor_postdeploy'].apply(classify_monitor)

# --- 3. Analysis ---
stats = analysis_df.groupby('dev_group')['is_monitored'].agg(['count', 'sum', 'mean'])
stats.columns = ['Total Systems', 'Monitored Systems', 'Monitoring Rate']

print("\n--- Monitoring Compliance by Development Ownership ---")
print(stats)

# Contingency Table
contingency = pd.crosstab(analysis_df['dev_group'], analysis_df['is_monitored'])
print("\n--- Contingency Table (0=No, 1=Yes) ---")
print(contingency)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

if p < 0.05:
    print("Result: Statistically significant difference detected.")
else:
    print("Result: No statistically significant difference detected.")

# --- 4. Visualization ---
plt.figure(figsize=(8, 6))

# Prepare data for plotting
groups = stats.index.tolist()
rates = stats['Monitoring Rate'].tolist()
colors = ['#e74c3c', '#3498db'] # Red for Contractor, Blue for In-house (usually)
if groups[0] == 'In-house':
    colors = ['#3498db', '#e74c3c']

bars = plt.bar(groups, rates, color=colors, alpha=0.8, edgecolor='black')

plt.title('AI System Continuous Monitoring Rates: In-house vs. Contractor')
plt.ylabel('Proportion of Systems with Monitoring')
plt.xlabel('Development Ownership')
plt.ylim(0, max(rates) * 1.2 if max(rates) > 0 else 0.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Annotate bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (max(rates)*0.02 if max(rates)>0 else 0.005),
             f'{height:.1%}',
             ha='center', va='bottom', fontweight='bold')

plt.show()