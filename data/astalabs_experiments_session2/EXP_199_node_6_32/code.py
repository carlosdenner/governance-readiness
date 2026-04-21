import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import sys

# Set the dataset path
dataset_path = "astalabs_discovery_all_data.csv"

# Load the dataset
print("Loading dataset...")
try:
    df = pd.read_csv(dataset_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: Dataset not found at {dataset_path}")
    sys.exit(1)

# Filter for EO13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 Scored records: {len(eo_data)}")

# Define target columns
agency_col = '3_agency'
transparency_col = '59_ai_notice'
accountability_col = '55_independent_eval'

# Updated Logic based on Debug Output
def check_transparency(val):
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    # Explicit negative cases
    if 'none of the above' in s: return 0
    if 'n/a' in s: return 0
    if 'waived' in s: return 0
    if 'not safety' in s: return 0
    
    # Positive indicators found in the dataset
    if 'online' in s: return 1
    if 'in-person' in s: return 1
    if 'in person' in s: return 1
    if 'email' in s: return 1
    if 'telephone' in s: return 1
    if 'other' in s: return 1
    
    return 0

def check_accountability(val):
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    # Positive indicators
    if s.startswith('yes'): return 1
    if s == 'true': return 1
    return 0

# Apply mapping
eo_data['is_transparent'] = eo_data[transparency_col].apply(check_transparency)
eo_data['is_accountable'] = eo_data[accountability_col].apply(check_accountability)

print(f"Total Transparent Systems: {eo_data['is_transparent'].sum()}")
print(f"Total Accountable Systems: {eo_data['is_accountable'].sum()}")

# Group by Agency
agency_stats = eo_data.groupby(agency_col).agg(
    system_count=('source_row_num', 'count'),
    transparency_rate=('is_transparent', 'mean'),
    accountability_rate=('is_accountable', 'mean')
).reset_index()

# Filter for agencies with > 10 systems
min_systems = 10
filtered_agencies = agency_stats[agency_stats['system_count'] > min_systems].copy()

print(f"\nAgencies with > {min_systems} systems: {len(filtered_agencies)}")
print(filtered_agencies[[agency_col, 'system_count', 'transparency_rate', 'accountability_rate']])

# Check for variance
x = filtered_agencies['transparency_rate']
y = filtered_agencies['accountability_rate']

if len(filtered_agencies) < 2:
    print("\nNot enough agencies to calculate correlation.")
elif np.std(x) == 0 or np.std(y) == 0:
    print("\nVariance is zero for one of the variables. Cannot calculate correlation.")
    print(f"Std Dev Transparency: {np.std(x)}")
    print(f"Std Dev Accountability: {np.std(y)}")
else:
    # Calculate Correlations
    pearson_corr, pearson_p = stats.pearsonr(x, y)
    spearman_corr, spearman_p = stats.spearmanr(x, y)

    print("\n--- Correlation Results ---")
    print(f"Pearson Correlation:  {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

    if pearson_p < 0.05:
        print("Result: Statistically significant correlation found.")
    else:
        print("Result: No statistically significant correlation found.")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, edgecolors='b', s=filtered_agencies['system_count']*2)
    
    # Add regression line
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--', label=f'Fit: y={m:.2f}x + {b:.2f}')

    # Label points (top 5 by count to avoid clutter)
    top_agencies = filtered_agencies.nlargest(5, 'system_count')
    for i, row in top_agencies.iterrows():
        plt.text(row['transparency_rate'], row['accountability_rate'], row[agency_col][:15]+'...', fontsize=8, alpha=0.9)

    plt.title('Agency Transparency (AI Notice) vs. Accountability (Indep. Eval)')
    plt.xlabel('Transparency Rate (Proportion of Systems with AI Notice)')
    plt.ylabel('Accountability Rate (Proportion of Systems with Indep. Eval)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
