import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# Normalize 'Sector of Deployment' and 'Known AI Technical Failure' columns
sector_col = [c for c in aiid_df.columns if 'Sector of Deployment' in c][0]
failure_col = [c for c in aiid_df.columns if 'Known AI Technical Failure' in c][0]
aiid_df = aiid_df.rename(columns={sector_col: 'Sector', failure_col: 'Failure'})

# --- 1. Define Sectors ---
def categorize_sector(val):
    if pd.isna(val):
        return None
    val = str(val).lower()
    if 'health' in val:
        return 'Healthcare'
    # Public administration, defense, law enforcement map to Government
    # We exclude 'health' to handle mixed cases by prioritizing healthcare or just distinguishing
    if any(x in val for x in ['public administration', 'defense', 'law enforcement', 'government']):
        return 'Government'
    return None

aiid_df['Derived_Sector'] = aiid_df['Sector'].apply(categorize_sector)

# Filter for target sectors
target_df = aiid_df[aiid_df['Derived_Sector'].isin(['Healthcare', 'Government'])].copy()
print(f"Rows found for Healthcare/Government: {len(target_df)}")
print(target_df['Derived_Sector'].value_counts())

# --- 2. Categorize Failures ---
def categorize_failure(val):
    if pd.isna(val):
        return 'Other'
    val = str(val).lower()
    
    # Define keywords
    unsafe_keywords = ['unsafe', 'control', 'robustness', 'reliability', 'system behavior']
    hci_keywords = ['human', 'operator', 'interaction', 'user', 'mistake', 'hci']
    
    is_unsafe = any(k in val for k in unsafe_keywords)
    is_hci = any(k in val for k in hci_keywords)
    
    if is_unsafe and is_hci:
        return 'Both'
    elif is_unsafe:
        return 'Unsafe System Behavior'
    elif is_hci:
        return 'Human-Computer Interaction'
    else:
        return 'Other'

target_df['Failure_Category'] = target_df['Failure'].apply(categorize_failure)

print("\n--- Failure Category Distribution ---")
print(target_df['Failure_Category'].value_counts())

# --- 3. Statistical Analysis ---
# We focus on the hypothesis: Healthcare -> Unsafe, Government -> HCI
# We will filter for just these two failure types to see the direct trade-off, 
# or use the full table to see independence.
# Let's use 'Unsafe System Behavior' and 'Human-Computer Interaction' categories.

analysis_df = target_df[target_df['Failure_Category'].isin(['Unsafe System Behavior', 'Human-Computer Interaction'])].copy()

if len(analysis_df) < 5:
    print("\nInsufficient data for specific failure comparison. showing full contingency.")
    contingency = pd.crosstab(target_df['Derived_Sector'], target_df['Failure_Category'])
else:
    contingency = pd.crosstab(analysis_df['Derived_Sector'], analysis_df['Failure_Category'])

print("\n--- Contingency Table (Target Categories) ---")
print(contingency)

# Chi-Square Test
if contingency.size > 0 and contingency.sum().sum() > 5:
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2: {chi2:.4f}, p-value: {p:.4e}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    # Calculate row-wise percentages for better comparison
    props = contingency.div(contingency.sum(axis=1), axis=0).reset_index()
    props_melted = props.melt(id_vars='Derived_Sector', var_name='Failure Type', value_name='Proportion')
    
    sns.barplot(data=props_melted, x='Derived_Sector', y='Proportion', hue='Failure Type')
    plt.title('Comparison of Failure Types by Sector')
    plt.ylabel('Proportion of Incidents (within filtered types)')
    plt.show()
else:
    print("Not enough data for statistical test.")
