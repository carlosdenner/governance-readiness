import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
# Note: Dataset files are one level above the current working directory
file_path = '../astalabs_discovery_all_data.csv'

try:
    # Using low_memory=False to avoid DtypeWarning mixing types
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running in a different environment structure
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for aiid_incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# 3. Select relevant columns and clean data
sector_col = 'Sector of Deployment'
failure_col = 'Known AI Technical Failure'

# Check if columns exist (using exact names from metadata or previous output)
# Metadata listed: 'Sector of Deployment', 'Known AI Technical Failure'
if sector_col not in aiid_df.columns or failure_col not in aiid_df.columns:
    print(f"Columns '{sector_col}' or '{failure_col}' not found. Available columns:")
    print(aiid_df.columns.tolist())
    exit(1)

# Drop NaN values in relevant columns for this analysis
subset = aiid_df[[sector_col, failure_col]].dropna()

# Clean strings
subset[sector_col] = subset[sector_col].astype(str).str.strip()
subset[failure_col] = subset[failure_col].astype(str).str.strip()

# 4. Filter for Top Sectors and Failures to ensure statistical relevance
# Select top N sectors by count
top_sectors_count = 8
top_sectors = subset[sector_col].value_counts().nlargest(top_sectors_count).index.tolist()

# Select top M failure types by count
# (AIID failure types can be sparse, so we focus on the most common ones)
top_failures_count = 8
top_failures = subset[failure_col].value_counts().nlargest(top_failures_count).index.tolist()

# Filter the dataframe
filtered_df = subset[
    (subset[sector_col].isin(top_sectors)) & 
    (subset[failure_col].isin(top_failures))
]

print(f"Analyzing top {top_sectors_count} sectors and top {top_failures_count} failure types.")
print(f"Data points after filtering: {len(filtered_df)}")

# 5. Create Contingency Table
contingency_table = pd.crosstab(filtered_df[sector_col], filtered_df[failure_col])

# 6. Chi-Square Test and Cramer's V
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Cramer's V calculation
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

print(f"\nCramer's V: {cramers_v:.4f}")
print(f"P-value: {p:.4e}")

# 7. Calculate Standardized Residuals
# Residual = (Observed - Expected) / sqrt(Expected)
# Adjusted Residual (Standardized) considers row/col totals, but Pearson residual is often sufficient for heatmaps.
# Here we calculate Pearson residuals for the heatmap visualization
pearson_residuals = (contingency_table - expected) / np.sqrt(expected)

# Identify dominant failure mode per sector (max residual)
dominant_failures = {}
for sector in contingency_table.index:
    sector_residuals = pearson_residuals.loc[sector]
    max_res_failure = sector_residuals.idxmax()
    max_res_val = sector_residuals.max()
    dominant_failures[sector] = (max_res_failure, max_res_val)

print("\nDominant Failure Mode per Sector (highest positive residual):")
for sector, (fail_type, res_val) in dominant_failures.items():
    print(f"  - {sector}: {fail_type} (Residual: {res_val:.2f})")

# 8. Visualization
plt.figure(figsize=(12, 8))

# Plotting Counts
plt.subplot(2, 1, 1)
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
plt.title(f'Contingency Table: Sector vs Technical Failure (Top {top_sectors_count}x{top_failures_count})')
plt.ylabel('Sector')
plt.xlabel('Technical Failure Type')

# Plotting Residuals (Associations)
plt.subplot(2, 1, 2)
# Use a diverging colormap to show positive (over-represented) and negative (under-represented) associations
sns.heatmap(pearson_residuals, annot=True, fmt='.2f', cmap='vlag', center=0)
plt.title('Pearson Residuals (Association Strength)')
plt.ylabel('Sector')
plt.xlabel('Technical Failure Type')

plt.tight_layout()
plt.show()
