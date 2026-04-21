import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Load dataset from current directory
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    print("Error: Dataset not found in current directory.")
    sys.exit(1)

# Filter for EO13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 Subset Shape: {eo_df.shape}")

# Define columns
col_impact = '17_impact_type'
col_eval = '55_independent_eval'

# Inspect unique values to ensure correct mapping
print(f"\nUnique values in '{col_impact}':\n{eo_df[col_impact].dropna().unique()}")
print(f"\nUnique values in '{col_eval}':\n{eo_df[col_eval].dropna().unique()}")

# 1. Create 'High Risk' (Rights-Impacting) flag
# We classify 'Rights-Impacting' and 'Both' (Rights + Safety) as High Risk.
def classify_risk(val):
    s = str(val).lower()
    if 'rights' in s or 'both' in s:
        return 'Rights-Impacting'
    return 'Other'

eo_df['risk_group'] = eo_df[col_impact].apply(classify_risk)

# 2. Create 'Evaluated' flag
# We look for affirmative 'Yes'
def classify_eval(val):
    s = str(val).lower()
    if s.startswith('yes'):
        return 'Evaluated'
    return 'Not Evaluated'

eo_df['eval_status'] = eo_df[col_eval].apply(classify_eval)

# 3. Generate Contingency Table
contingency = pd.crosstab(eo_df['risk_group'], eo_df['eval_status'])
print("\nContingency Table (Counts):")
print(contingency)

# Check if we have enough data for statistical testing
if contingency.shape[0] < 2 or contingency.shape[1] < 2:
    print("\nInsufficient data variation for statistical testing.")
else:
    # 4. Perform Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    # Interpret
    alpha = 0.05
    if p < alpha:
        print("Result: Significant difference in evaluation rates found (Null Hypothesis Rejected).")
    else:
        print("Result: No significant difference in evaluation rates found (Supporting the 'Compliance Paradox').")

    # 5. Calculate Proportions for Plotting
    # Normalize by index (row) to get % evaluated within each risk group
    props = pd.crosstab(eo_df['risk_group'], eo_df['eval_status'], normalize='index') * 100
    print("\nEvaluation Rates (%):")
    print(props)

    # Plot
    if 'Evaluated' in props.columns:
        plt.figure(figsize=(8, 6))
        ax = props['Evaluated'].plot(kind='bar', color=['#1f77b4', '#d62728'], rot=0)
        plt.title('Percentage of AI Systems with Independent Evaluation by Risk Group')
        plt.ylabel('% Independently Evaluated')
        plt.xlabel('Risk Classification')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for p_rect in ax.patches:
            height = p_rect.get_height()
            ax.annotate(f'{height:.1f}%', 
                        (p_rect.get_x() + p_rect.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=10, weight='bold')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No 'Evaluated' systems found to plot.")