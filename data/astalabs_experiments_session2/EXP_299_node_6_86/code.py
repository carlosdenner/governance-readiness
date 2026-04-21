import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
print(f"Loading dataset from {filename}...")

try:
    df = pd.read_csv(filename, low_memory=False)
except FileNotFoundError:
    # Fallback to relative path if needed
    df = pd.read_csv(f'../{filename}', low_memory=False)

# Filter for EO 13960 Scored data
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {subset.shape}")

# --- Step 1: Define COTS vs GOTS groups ---
# Using '22_dev_method' as proxy
# 'Developed in-house.' -> Government (GOTS)
# 'Developed with contracting resources.' -> Commercial (COTS)

def classify_source(val):
    s = str(val).lower()
    if 'in-house' in s and 'contracting' not in s:
        return 'Government (GOTS)'
    elif 'contracting' in s and 'in-house' not in s:
        return 'Commercial (COTS)'
    else:
        return None

subset['procurement_type'] = subset['22_dev_method'].apply(classify_source)

# Filter to only these two groups
analysis_df = subset.dropna(subset=['procurement_type']).copy()
print(f"Analyzable subset (COTS vs GOTS) shape: {analysis_df.shape}")
print("Group distribution:")
print(analysis_df['procurement_type'].value_counts())

# --- Step 2: Binarize Transparency Columns ---

# Variable 1: Code Access ('38_code_access')
# Logic: Contains 'Yes' -> 1, Contains 'No' -> 0
def parse_code_access(val):
    s = str(val).lower()
    if 'yes' in s:
        return 1
    elif 'no' in s:
        return 0
    return np.nan  # Treat unclear/nan as missing

analysis_df['has_code_access'] = analysis_df['38_code_access'].apply(parse_code_access)

# Variable 2: Data Documentation ('34_data_docs')
# Logic: 'missing', 'no' -> 0; 'complete', 'partial', 'available', 'yes' -> 1
def parse_data_docs(val):
    s = str(val).lower()
    if pd.isna(val) or s == 'nan' or 'not reported' in s:
        return np.nan
    if 'missing' in s or 'not available' in s or s == 'no':
        return 0
    if 'complete' in s or 'partial' in s or 'available' in s or 'yes' in s:
        return 1
    return 0 # Default fallback for negatives not caught, though risky. Let's inspect coverage.

analysis_df['has_data_docs'] = analysis_df['34_data_docs'].apply(parse_data_docs)

# --- Step 3: Statistical Analysis ---

results = []
metrics = [('has_code_access', 'Code Access'), ('has_data_docs', 'Data Documentation')]

print("\n--- Statistical Tests ---")

for col, label in metrics:
    # Drop NaNs for the specific test
    valid_data = analysis_df.dropna(subset=[col])
    
    # Contingency Table
    contingency = pd.crosstab(valid_data['procurement_type'], valid_data[col])
    
    # Chi-Square Test
    chi2, p, dof, ex = chi2_contingency(contingency)
    
    # Calculate Rates
    rates = valid_data.groupby('procurement_type')[col].mean()
    gots_rate = rates.get('Government (GOTS)', 0)
    cots_rate = rates.get('Commercial (COTS)', 0)
    
    results.append({
        'Metric': label,
        'GOTS Rate': gots_rate,
        'COTS Rate': cots_rate,
        'p-value': p,
        'Significant': p < 0.05
    })
    
    print(f"\nMetric: {label}")
    print(contingency)
    print(f"GOTS Rate: {gots_rate:.2%}, COTS Rate: {cots_rate:.2%}")
    print(f"Chi-Square: {chi2:.4f}, p-value: {p:.4e}")

# --- Step 4: Visualization ---

res_df = pd.DataFrame(results)

# Plotting
labels = res_df['Metric']
gots_means = res_df['GOTS Rate'] * 100
cots_means = res_df['COTS Rate'] * 100

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, gots_means, width, label='Government (GOTS)', color='#1f77b4')
rects2 = ax.bar(x + width/2, cots_means, width, label='Commercial (COTS)', color='#ff7f0e')

ax.set_ylabel('Transparency Rate (%)')
ax.set_title('Transparency Gap: Commercial vs Government AI')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add significance stars
for i, p_val in enumerate(res_df['p-value']):
    if p_val < 0.001:
        sig = '***'
    elif p_val < 0.01:
        sig = '**'
    elif p_val < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    # Height for annotation
    max_h = max(gots_means[i], cots_means[i])
    ax.text(i, max_h + 2, sig, ha='center', va='bottom', fontweight='bold')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, 110)
plt.tight_layout()
plt.show()
