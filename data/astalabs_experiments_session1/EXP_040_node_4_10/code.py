import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import os

# Define the filename
filename = 'step3_incident_coding.csv'

# Attempt to load the dataset, handling potential path issues
if os.path.exists(filename):
    df = pd.read_csv(filename)
elif os.path.exists(f'../{filename}'):
    df = pd.read_csv(f'../{filename}')
else:
    print(f"Error: {filename} not found in current or parent directory.")
    sys.exit(1)

# Preprocess the data
# Normalize the grouping column
df['split_normalized'] = df['trust_integration_split'].astype(str).str.lower().str.strip()

# Create the Complexity Group variable
# 'Both' implies Multi-Domain; anything else (Trust-Dominant, Integration-Dominant) is Single-Domain
df['domain_complexity'] = df['split_normalized'].apply(
    lambda x: 'Multi-Domain' if 'both' in x else 'Single-Domain'
)

# Extract the technique counts for each group
group_multi = df[df['domain_complexity'] == 'Multi-Domain']['technique_count'].dropna()
group_single = df[df['domain_complexity'] == 'Single-Domain']['technique_count'].dropna()

# 1. Descriptive Statistics
print("=== Descriptive Statistics: Technique Count by Domain Complexity ===")
stats_df = df.groupby('domain_complexity')['technique_count'].describe()
print(stats_df)
print("\n")

# 2. Statistical Test (Welch's t-test)
# We use equal_var=False because sample sizes are likely unequal (Metadata suggests 46 vs 6 split)
t_stat, p_val = stats.ttest_ind(group_multi, group_single, equal_var=False)

print("=== Welch's T-Test Results ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value:     {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Conclusion: The difference is statistically significant (Reject H0).")
else:
    print("Conclusion: The difference is NOT statistically significant (Fail to Reject H0).")

# 3. Visualization
plt.figure(figsize=(8, 6))
# Using a boxplot for clear comparison of medians and spread
plt.boxplot([group_multi, group_single], labels=['Multi-Domain', 'Single-Domain'], patch_artist=True,
            boxprops=dict(facecolor="lightblue"))
plt.title('Distribution of Attack Technique Counts by Competency Gap Complexity')
plt.ylabel('Count of ATLAS Techniques Used')
plt.xlabel('Competency Domain Gap')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()