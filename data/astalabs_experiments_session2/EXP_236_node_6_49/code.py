import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running in a different environment structure
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()

# Helper function to parse list-like strings
def parse_list_col(val):
    if pd.isna(val):
        return []
    if '|' in str(val):
        return [x.strip() for x in str(val).split('|') if x.strip()]
    return [str(val).strip()]

# Calculate technique counts
# The 'techniques' column follows the same pipe-separated format
atlas_df['technique_count'] = atlas_df['techniques'].apply(lambda x: len(set(parse_list_col(x))))

# Identify 'Impact' cases
# Debugging showed format is like {{impact.id}}
atlas_df['has_impact'] = atlas_df['tactics'].apply(lambda x: 'impact.id' in str(x).lower() if pd.notna(x) else False)

# Split groups
impact_counts = atlas_df[atlas_df['has_impact'] == True]['technique_count']
no_impact_counts = atlas_df[atlas_df['has_impact'] == False]['technique_count']

print(f"Group 'Impact': n={len(impact_counts)}, Mean={impact_counts.mean():.2f}, Median={impact_counts.median()}")
print(f"Group 'Non-Impact': n={len(no_impact_counts)}, Mean={no_impact_counts.mean():.2f}, Median={no_impact_counts.median()}")

# Statistical Test
# Mann-Whitney U is safer for small sample sizes and non-normal distributions
stat, p_val = stats.mannwhitneyu(impact_counts, no_impact_counts, alternative='two-sided')
print(f"\nMann-Whitney U Test: U={stat}, p-value={p_val:.4f}")

t_stat, t_p_val = stats.ttest_ind(impact_counts, no_impact_counts, equal_var=False)
print(f"Welch's T-Test: t={t_stat:.4f}, p-value={t_p_val:.4f}")

if p_val < 0.05:
    print("\nResult: Statistically significant difference found.")
else:
    print("\nResult: No statistically significant difference found.")

# Visualization
plt.figure(figsize=(8, 6))
data_to_plot = [no_impact_counts, impact_counts]
plt.boxplot(data_to_plot, tick_labels=['Non-Impact', 'Impact'])
plt.title('Technique Complexity: Impact vs. Non-Impact Cases')
plt.ylabel('Number of Unique Techniques')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()