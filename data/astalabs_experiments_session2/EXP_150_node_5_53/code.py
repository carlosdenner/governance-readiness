import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Total EO 13960 records: {len(eo_data)}")

# --- PIVOT: 22_dev_method contains Sourcing info, not Agile/Waterfall ---
# New Objective: Compare In-House vs. Contractor Sourcing on Monitoring Rates

# 1. Categorize Sourcing (formerly '22_dev_method')
sourcing_col = '22_dev_method'

def categorize_sourcing(val):
    text = str(val).lower().strip()
    if 'in-house' in text and 'contracting' not in text:
        return 'In-House'
    elif 'contracting' in text and 'in-house' not in text:
        return 'Contractor'
    elif 'both' in text or ('in-house' in text and 'contracting' in text):
        return 'Hybrid'
    else:
        return 'Unknown'

eo_data['sourcing_category'] = eo_data[sourcing_col].apply(categorize_sourcing)
print("\nSourcing Category Distribution:")
print(eo_data['sourcing_category'].value_counts())

# 2. Categorize Monitoring (56_monitor_postdeploy)
monitor_col = '56_monitor_postdeploy'

def categorize_monitoring(val):
    if pd.isna(val):
        return 'Unknown'
    text = str(val).lower().strip()
    
    # Positive indicators (Active monitoring)
    if any(x in text for x in ['intermittent', 'automated', 'established process', 'regularly scheduled']):
        return 'Monitored'
    # Negative indicators (No monitoring)
    elif 'no monitoring' in text:
        return 'Not Monitored'
    else:
        return 'Unknown'

eo_data['monitoring_status'] = eo_data[monitor_col].apply(categorize_monitoring)
print("\nMonitoring Status Distribution:")
print(eo_data['monitoring_status'].value_counts())

# 3. Create Analysis Set (Filter out Unknowns)
analysis_set = eo_data[
    (eo_data['sourcing_category'].isin(['In-House', 'Contractor'])) &
    (eo_data['monitoring_status'].isin(['Monitored', 'Not Monitored']))
].copy()

print(f"\nRecords with valid Sourcing AND Monitoring data: {len(analysis_set)}")

# 4. Statistical Analysis
if len(analysis_set) > 0:
    # Convert to binary for mean calculation
    analysis_set['is_monitored_bool'] = (analysis_set['monitoring_status'] == 'Monitored').astype(int)
    
    # Group stats
    in_house = analysis_set[analysis_set['sourcing_category'] == 'In-House']
    contractor = analysis_set[analysis_set['sourcing_category'] == 'Contractor']
    
    rate_in_house = in_house['is_monitored_bool'].mean()
    rate_contractor = contractor['is_monitored_bool'].mean()
    
    print(f"\nIn-House Monitoring Rate: {rate_in_house:.2%} (n={len(in_house)})")
    print(f"Contractor Monitoring Rate: {rate_contractor:.2%} (n={len(contractor)})")
    
    # Contingency Table
    contingency = pd.crosstab(analysis_set['sourcing_category'], analysis_set['monitoring_status'])
    print("\nContingency Table:")
    print(contingency)
    
    # Chi-Square Test
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test: p-value = {p:.5f}")
    
    if p < 0.05:
        print("Result: Statistically Significant Difference")
    else:
        print("Result: No Statistically Significant Difference")
        
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.bar(['In-House', 'Contractor'], [rate_in_house, rate_contractor], color=['#4CAF50', '#FF9800'])
    plt.title('AI Monitoring Rates by Sourcing Method')
    plt.ylabel('Proportion Monitored')
    plt.ylim(0, 1.1)
    
    for i, v in enumerate([rate_in_house, rate_contractor]):
        plt.text(i, v + 0.02, f"{v:.1%}", ha='center')
        
    plt.show()
else:
    print("\nInsufficient overlapping data to perform statistical test.")
