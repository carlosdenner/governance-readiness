import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"EO 13960 Data Shape: {eo_data.shape}")

# Target columns
dev_col = '22_dev_method'
access_col = '38_code_access'

# Check if columns exist
if dev_col not in eo_data.columns or access_col not in eo_data.columns:
    print(f"Columns '{dev_col}' or '{access_col}' not found.")
else:
    # 1. Clean Development Method
    def clean_dev_method(val):
        if pd.isna(val):
            return None
        v = str(val).lower()
        # Updated mapping based on dataset values
        if 'contracting resources' in v:
            return 'Commercial/Contractor'
        if 'in-house' in v and 'contracting' not in v: # Strict in-house
            return 'Government/In-House'
        return None

    eo_data['procurement_type'] = eo_data[dev_col].apply(clean_dev_method)

    # 2. Clean Code Access
    def clean_access(val):
        if pd.isna(val):
            return None
        v = str(val).lower().strip()
        if v.startswith('no') or 'not have access' in v:
            return 'No Access'
        if 'yes' in v:
            return 'Access Granted'
        return None

    eo_data['code_access_status'] = eo_data[access_col].apply(clean_access)

    # Filter dataset for analysis
    analysis_df = eo_data.dropna(subset=['procurement_type', 'code_access_status'])

    print(f"\nRows available for analysis: {len(analysis_df)}")
    print(f"Breakdown by Procurement Type:\n{analysis_df['procurement_type'].value_counts()}")

    if len(analysis_df['procurement_type'].unique()) > 1:
        # Contingency Table
        contingency_table = pd.crosstab(analysis_df['procurement_type'], analysis_df['code_access_status'])
        print("\nContingency Table (Count):")
        print(contingency_table)
        
        # Percentage Table
        contingency_pct = pd.crosstab(analysis_df['procurement_type'], analysis_df['code_access_status'], normalize='index') * 100
        print("\nContingency Table (Percentage):")
        print(contingency_pct.round(2))

        # Chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nChi-Square Test Results:")
        print(f"Chi2 Stat: {chi2:.4f}")
        print(f"P-value: {p:.4e}")
        
        # Interpretation
        alpha = 0.05
        if p < alpha:
            print("\nResult: Statistically significant relationship found.")
            # Check direction
            try:
                comm_access = contingency_pct.loc['Commercial/Contractor', 'Access Granted']
                gov_access = contingency_pct.loc['Government/In-House', 'Access Granted']
                print(f"Commercial/Contractor Access Rate: {comm_access:.2f}%")
                print(f"Government/In-House Access Rate: {gov_access:.2f}%")
                
                if comm_access < gov_access:
                    print(f"Conclusion: Commercial systems are significantly LESS likely to grant access ({comm_access:.1f}% vs {gov_access:.1f}%), supporting the hypothesis.")
                else:
                    print(f"Conclusion: Commercial systems are MORE or EQUALLY likely to grant access, rejecting the hypothesis direction.")
            except KeyError:
                print("Could not determine directionality due to missing keys in pivot.")
        else:
            print("\nResult: No statistically significant relationship found.")
            
        # Visualization
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency_pct, annot=True, fmt='.1f', cmap='RdBu', cbar_kws={'label': 'Percentage'})
        plt.title('Code Access: Commercial vs. In-House Development')
        plt.ylabel('Procurement Type')
        plt.xlabel('Code Access Status')
        plt.show()
    else:
        print("Insufficient data: Only one procurement type found after filtering.")