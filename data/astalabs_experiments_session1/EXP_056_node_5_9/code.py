import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# [debug] Load dataset to inspect columns and values
# df_debug = pd.read_csv('../step2_competency_statements.csv')
# print(df_debug.head())
# print(df_debug['req_id'].unique())
# print(df_debug['confidence'].unique())

# 1. Load the dataset
file_path = '../step2_competency_statements.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    # Fallback for local testing if needed, though instructions say one level up
    df = pd.read_csv('step2_competency_statements.csv')

# 2. Extract Framework Source
def extract_source(req_id):
    if isinstance(req_id, str):
        if req_id.startswith('NIST'):
            return 'NIST'
        elif req_id.startswith('EU'):
            return 'EU'
        elif req_id.startswith('OWASP'):
            return 'OWASP'
    return 'Other'

df['source'] = df['req_id'].apply(extract_source)

# Filter out 'Other' if any (though metadata suggests these are the main ones)
df = df[df['source'] != 'Other']

# 3. Convert Confidence to Numeric
confidence_map = {'High': 3, 'Medium': 2, 'Low': 1, 'high': 3, 'medium': 2, 'low': 1}
df['confidence_score'] = df['confidence'].map(confidence_map)

# Remove rows with NaN confidence if any
df = df.dropna(subset=['confidence_score'])

# 4. Descriptive Statistics
group_stats = df.groupby('source')['confidence_score'].agg(['count', 'mean', 'std', 'sem'])
print("=== Descriptive Statistics by Source ===")
print(group_stats)
print("\n")

# 5. Statistical Test (One-Way ANOVA)
# Extract groups
groups = [df[df['source'] == s]['confidence_score'].values for s in ['NIST', 'EU', 'OWASP']]

# Check if we have enough data in each group
if all(len(g) > 1 for g in groups):
    f_stat, p_value = stats.f_oneway(*groups)
    
    print("=== One-Way ANOVA Results ===")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: Statistically Significant (p < 0.05)")
        # Post-hoc test (Tukey HSD) if significant
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tukey = pairwise_tukeyhsd(endog=df['confidence_score'], groups=df['source'], alpha=0.05)
            print("\n=== Tukey HSD Post-hoc Test ===")
            print(tukey)
        except ImportError:
            print("statsmodels not installed, skipping Tukey HSD.")
    else:
        print("Result: Not Statistically Significant (p >= 0.05)")
else:
    print("Insufficient data in one or more groups to perform ANOVA.")

# 6. Visualization
plt.figure(figsize=(10, 6))
bars = plt.bar(group_stats.index, group_stats['mean'], 
               yerr=group_stats['sem'], capsize=10, 
               color=['#4c72b0', '#55a868', '#c44e52'], alpha=0.8)

plt.title('Mean Confidence Score of Competency Statements by Source Framework')
plt.xlabel('Framework Source')
plt.ylabel('Mean Confidence Score (1=Low, 3=High)')
plt.ylim(0, 3.5)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()