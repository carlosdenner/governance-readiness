import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

print("Starting Experiment: Vendor-Governance Gap Analysis (Revised)")

# 1. Load Data
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# 2. Filter for EO13960 Scored Data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Total EO13960 Records: {len(eo_data)}")

# 3. Clean Source Type ('22_dev_method')
# Map to 'In-House' vs 'Contractor'
def categorize_source(val):
    if pd.isna(val):
        return None
    val = str(val).strip()
    if 'Developed in-house' in val:
        return 'In-House (GOTS)'
    elif 'Developed with contracting resources' in val:
        return 'Contractor (COTS/Vendor)'
    return None

eo_data['source_type'] = eo_data['22_dev_method'].apply(categorize_source)

# 4. Clean Impact Assessment ('52_impact_assessment')
# Filter out NaNs to analyze only reported data
valid_impact_data = eo_data.dropna(subset=['52_impact_assessment']).copy()

def check_compliance(val):
    val = str(val).lower().strip()
    if val == 'yes':
        return 1
    # Treat 'no' and 'planned' as 0 (not currently compliant)
    return 0

valid_impact_data['has_impact_assessment'] = valid_impact_data['52_impact_assessment'].apply(check_compliance)

# 5. Analysis: Intersection of valid Source and valid Impact Assessment
analysis_df = valid_impact_data.dropna(subset=['source_type'])

print(f"Records with valid Source AND Impact Assessment data: {len(analysis_df)}")
print(analysis_df['source_type'].value_counts())

if len(analysis_df) < 5:
    print("Insufficient data for statistical analysis.")
else:
    # Compliance Rates
    compliance_rates = analysis_df.groupby('source_type')['has_impact_assessment'].agg(['mean', 'count', 'sum'])
    compliance_rates['percentage'] = compliance_rates['mean'] * 100
    print("\nCompliance Rates by Source Type:")
    print(compliance_rates)

    # Contingency Table
    contingency_table = pd.crosstab(analysis_df['source_type'], analysis_df['has_impact_assessment'])
    print("\nContingency Table (0=No/Planned, 1=Yes):")
    print(contingency_table)

    # Chi-Square Test
    # Check frequency assumption
    if (contingency_table < 5).any().any():
        print("\nWarning: Low cell counts (<5) detected. Using Fisher's Exact Test instead of Chi-Square.")
        odds_ratio, p_value = stats.fisher_exact(contingency_table)
        test_name = "Fisher's Exact Test"
        stat_val = odds_ratio
    else:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        test_name = "Chi-Square Test"
        stat_val = chi2

    print(f"\n{test_name} Results:")
    print(f"Statistic: {stat_val:.4f}")
    print(f"P-Value: {p_value:.4f}")

    # Visualization
    plt.figure(figsize=(8, 6))
    colors = ['#4C72B0', '#55A868'] # Muted blue and green
    bars = plt.bar(compliance_rates.index, compliance_rates['percentage'], color=colors)
    plt.title('Impact Assessment Compliance by Development Source')
    plt.ylabel('Compliance Rate (%)')
    plt.xlabel('Source')
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}% (n={int(compliance_rates.loc[compliance_rates["percentage"] == height*10]["count"].values[0]) if not compliance_rates[compliance_rates["percentage"] == height].empty else ""})',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
