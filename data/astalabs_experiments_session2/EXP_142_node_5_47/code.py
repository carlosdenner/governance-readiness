import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    print("File not found at '../astalabs_discovery_all_data.csv', trying local directory...")
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print("Dataset filtered. EO 13960 records:", len(eo_data))

# Inspect columns of interest
dev_col = '22_dev_method'
appeal_col = '65_appeal_process'

# Data Cleaning & Mapping

# Corrected Mapping Function based on previous output unique values
def map_dev_method(val):
    if pd.isna(val):
        return None
    val = str(val).lower().strip()
    
    # Check for hybrid first to exclude it
    if 'both' in val:
        return None
        
    if 'contract' in val:
        return 'Contractor'
    elif 'in-house' in val or 'government' in val:
        return 'In-house'
    
    return None

eo_data['dev_category'] = eo_data[dev_col].apply(map_dev_method)

# Map Appeal Process
def map_appeal(val):
    if pd.isna(val):
        return 'No'
    val = str(val).lower().strip()
    if val == 'yes':
        return 'Yes'
    return 'No'

eo_data['has_appeal'] = eo_data[appeal_col].apply(map_appeal)

# Filter for valid development categories
analysis_df = eo_data.dropna(subset=['dev_category'])

print(f"\nRecords after filtering for valid development method (Contractor vs In-house): {len(analysis_df)}")
print("Distribution by Category:")
print(analysis_df['dev_category'].value_counts())

# Contingency Table
contingency_table = pd.crosstab(analysis_df['dev_category'], analysis_df['has_appeal'])
print("\nContingency Table (Appeal Process vs Dev Method):")
print(contingency_table)

# Chi-square Test
if contingency_table.shape[0] == 2:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
else:
    print("\nError: Contingency table does not have 2 rows. Cannot perform Chi-square test properly.")

# Calculate percentages
props = pd.crosstab(analysis_df['dev_category'], analysis_df['has_appeal'], normalize='index') * 100
print("\nProportions (%):")
print(props)

# Visualization
if not props.empty:
    try:
        ax = props.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#d62728', '#2ca02c'])
        plt.title('Presence of Appeal Process by Development Method')
        plt.xlabel('Development Method')
        plt.ylabel('Percentage')
        plt.legend(title='Has Appeal Process', loc='upper right', labels=['No', 'Yes'])
        plt.xticks(rotation=0)
        
        # Add value labels
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white')
            
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")
else:
    print("Insufficient data for plotting.")