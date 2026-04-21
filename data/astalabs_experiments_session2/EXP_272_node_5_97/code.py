import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

# [debug]
print("Starting experiment: Justice Sector Transparency Deficit")

# 1. Load Data
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset not found in parent directory. Trying current directory.")
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 records found: {len(eo_data)}")

# 2. Categorize Agencies
# Define keyword lists for categorization based on Agency Name (3_agency)
justice_security_keywords = [
    'Justice', 'Homeland Security', 'Defense', 'State', 'Intelligence', 
    'Investigation', 'Prisons', 'Police'
]
health_services_keywords = [
    'Health', 'Human Services', 'Education', 'Social Security', 'Veterans', 
    'Labor', 'Housing', 'Agriculture', 'Interior'
]

def categorize_agency(agency_name):
    if pd.isna(agency_name):
        return 'Other'
    agency_str = str(agency_name).lower()
    
    # Check Justice/Security
    for kw in justice_security_keywords:
        if kw.lower() in agency_str:
            return 'Justice/Security'
            
    # Check Health/Services
    for kw in health_services_keywords:
        if kw.lower() in agency_str:
            return 'Health/Services'
            
    return 'Other'

eo_data['Sector_Group'] = eo_data['3_agency'].apply(categorize_agency)

# Filter out 'Other'
analysis_df = eo_data[eo_data['Sector_Group'].isin(['Justice/Security', 'Health/Services'])].copy()
print("\nGroup Sizes:")
print(analysis_df['Sector_Group'].value_counts())

# 3. Clean Target Columns
# Convert to boolean/numeric (1 for Yes, 0 for No/Other)
def clean_boolean_col(val):
    if pd.isna(val):
        return 0
    if str(val).strip().lower() == 'yes':
        return 1
    return 0

target_cols = {
    '59_ai_notice': 'Public Notice',
    '65_appeal_process': 'Appeal Process'
}

results = {}

for col, label in target_cols.items():
    analysis_df[col + '_clean'] = analysis_df[col].apply(clean_boolean_col)
    
    # Group statistics
    stats = analysis_df.groupby('Sector_Group')[col + '_clean'].agg(['sum', 'count', 'mean'])
    stats['percentage'] = stats['mean'] * 100
    results[label] = stats

    print(f"\n--- {label} Statistics ---")
    print(stats[['count', 'sum', 'percentage']])

# 4. Statistical Tests (Z-test)
print("\n--- Statistical Significance (Z-Test) ---")

sig_results = []

for col, label in target_cols.items():
    col_clean = col + '_clean'
    
    # Extract counts and nobs for the two groups
    group_stats = analysis_df.groupby('Sector_Group')[col_clean].agg(['sum', 'count'])
    
    # Ensure we have both groups
    if len(group_stats) != 2:
        print(f"Skipping {label}: Insufficient groups.")
        continue
        
    count = np.array([group_stats.loc['Justice/Security', 'sum'], group_stats.loc['Health/Services', 'sum']])
    nobs = np.array([group_stats.loc['Justice/Security', 'count'], group_stats.loc['Health/Services', 'count']])
    
    stat, pval = proportions_ztest(count, nobs, alternative='two-sided')
    
    print(f"{label}:")
    print(f"  Justice/Security Rate: {group_stats.loc['Justice/Security', 'sum']/group_stats.loc['Justice/Security', 'count']:.2%}")
    print(f"  Health/Services Rate:  {group_stats.loc['Health/Services', 'sum']/group_stats.loc['Health/Services', 'count']:.2%}")
    print(f"  Z-score: {stat:.4f}")
    print(f"  P-value: {pval:.4e}")
    sig_results.append({'Metric': label, 'P-value': pval, 'Significant': pval < 0.05})

# 5. Visualization
labels = list(target_cols.values())
justice_means = [analysis_df[analysis_df['Sector_Group']=='Justice/Security'][col + '_clean'].mean() for col in target_cols.keys()]
health_means = [analysis_df[analysis_df['Sector_Group']=='Health/Services'][col + '_clean'].mean() for col in target_cols.keys()]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, justice_means, width, label='Justice/Security', color='#d62728')
rects2 = ax.bar(x + width/2, health_means, width, label='Health/Services', color='#1f77b4')

ax.set_ylabel('Adoption Rate')
ax.set_title('Transparency Controls by Sector (EO 13960)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()
