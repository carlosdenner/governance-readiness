import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import sys

# Load the dataset
file_name = 'step3_incident_coding.csv'
try:
    df = pd.read_csv(file_name)
    print(f"Successfully loaded {file_name} with shape {df.shape}")
except FileNotFoundError:
    print(f"Error: {file_name} not found.")
    sys.exit(1)

# Normalize the column to handle potential casing issues
if 'trust_integration_split' in df.columns:
    # Print original unique values for debugging
    print("Original unique values in 'trust_integration_split':")
    print(df['trust_integration_split'].unique())
    
    # Normalize to lowercase and strip whitespace
    df['trust_integration_split_norm'] = df['trust_integration_split'].astype(str).str.lower().str.strip()
else:
    print("Column 'trust_integration_split' not found.")
    sys.exit(1)

# Define target groups (normalized)
target_groups = ['integration-dominant', 'trust-dominant']
subset = df[df['trust_integration_split_norm'].isin(target_groups)].copy()

# Display counts to verify sample size
print("\n--- Sample Counts (Normalized) ---")
counts = subset['trust_integration_split_norm'].value_counts()
print(counts)

# Extract vectors for statistical testing
integration_scores = subset[subset['trust_integration_split_norm'] == 'integration-dominant']['technique_count']
trust_scores = subset[subset['trust_integration_split_norm'] == 'trust-dominant']['technique_count']

# Descriptive Statistics
print("\n--- Descriptive Statistics for Technique Count ---")
if not subset.empty:
    desc_stats = subset.groupby('trust_integration_split_norm')['technique_count'].describe()
    print(desc_stats)
else:
    print("No data found for the specified groups after normalization.")

# Statistical Testing
print("\n--- Statistical Test Results ---")
if len(integration_scores) > 1 and len(trust_scores) > 1:
    # Welch's T-test (does not assume equal variance)
    t_stat, p_val = stats.ttest_ind(integration_scores, trust_scores, equal_var=False)
    print(f"Welch's T-test: Statistic={t_stat:.4f}, p-value={p_val:.4f}")
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_p_val = stats.mannwhitneyu(integration_scores, trust_scores)
    print(f"Mann-Whitney U test: Statistic={u_stat:.4f}, p-value={u_p_val:.4f}")
    
    alpha = 0.05
    if p_val < alpha:
        print("Result: Statistically significant difference found (p < 0.05).")
    else:
        print("Result: No statistically significant difference found (p >= 0.05).")
else:
    print("Insufficient sample size for statistical testing (need at least 2 per group).")
    print(f"Integration-Dominant count: {len(integration_scores)}")
    print(f"Trust-Dominant count: {len(trust_scores)}")

# Visualization
if not subset.empty:
    plt.figure(figsize=(8, 6))
    # Map normalized names back to Title Case for display
    subset['Display Label'] = subset['trust_integration_split_norm'].map({
        'integration-dominant': 'Integration-Dominant',
        'trust-dominant': 'Trust-Dominant'
    })
    
    # Boxplot to show distribution
    sns.boxplot(x='Display Label', y='technique_count', data=subset, palette="Set2", order=['Trust-Dominant', 'Integration-Dominant'])
    # Swarmplot to show individual data points
    sns.swarmplot(x='Display Label', y='technique_count', data=subset, color=".25", size=8, order=['Trust-Dominant', 'Integration-Dominant'])
    
    plt.title('Attack Technique Count by Competency Gap Type')
    plt.ylabel('Number of Distinct Techniques')
    plt.xlabel('Competency Gap Dominance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
