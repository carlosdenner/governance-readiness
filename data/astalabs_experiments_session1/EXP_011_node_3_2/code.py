import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the dataset
# The user specified that datasets are one level above the current working directory
file_path = '../step3_mitigation_gaps.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
except FileNotFoundError:
    # Fallback to current directory if the ../ path fails (just in case the environment differs)
    df = pd.read_csv('step3_mitigation_gaps.csv')
    print("Loaded step3_mitigation_gaps.csv from current directory")

# 2. Identify unique values in 'category'
categories = df['category'].unique()
print(f"\nUnique Categories Identified: {categories}")

# 3. Group by 'category' and calculate mean 'incident_count'
# We'll calculate count, mean, and std to understand the distribution
grouped_stats = df.groupby('category')['incident_count'].agg(['count', 'mean', 'std', 'min', 'max'])
print("\nIncident Count Statistics by Category:")
print(grouped_stats)

# 4. Perform One-Way ANOVA
# Prepare the data for ANOVA: list of arrays, one for each category
groups = [df[df['category'] == cat]['incident_count'] for cat in categories]

# We need at least two groups to perform ANOVA
if len(groups) > 1:
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA Results:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    
    alpha = 0.05
    if p_val < alpha:
        print("Result: Statistically significant difference found between categories (Reject H0).")
    else:
        print("Result: No statistically significant difference found between categories (Fail to reject H0).")
else:
    print("\nInsufficient categories for ANOVA.")

# Visualization: Boxplot to show the distribution variance
plt.figure(figsize=(10, 6))
# Create a boxplot grouping by category
# Using pandas plotting directly for convenience
df.boxplot(column='incident_count', by='category', grid=True)
plt.title('Distribution of Incident Counts by Mitigation Category')
plt.suptitle('')  # Removes the default pandas suptitle
plt.ylabel('Incident Count')
plt.xlabel('Category')
plt.show()