import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define file path based on instruction
file_path = '../step3_incident_coding.csv'

# Check if file exists there, otherwise try current directory as fallback
if not os.path.exists(file_path):
    file_path = 'step3_incident_coding.csv'

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# Normalize the split column to handle case sensitivity and whitespace
df['split_clean'] = df['trust_integration_split'].astype(str).str.lower().str.strip()

# Create the binary grouping: 'Both' vs 'Distinct'
df['group'] = df['split_clean'].apply(lambda x: 'Both' if x == 'both' else 'Distinct')

# Extract technique counts for each group
both_counts = df[df['group'] == 'Both']['technique_count']
distinct_counts = df[df['group'] == 'Distinct']['technique_count']

# Print group sizes
print("\nGroup Sample Sizes:")
print(f"Both: {len(both_counts)}")
print(f"Distinct: {len(distinct_counts)}")

# Descriptive Statistics
print("\nDescriptive Statistics for Technique Count:")
print(df.groupby('group')['technique_count'].describe())

# T-test (Using Welch's t-test due to likely unequal sample sizes and variance)
t_stat, p_val = stats.ttest_ind(both_counts, distinct_counts, equal_var=False)

print("\nIndependent Samples T-Test (Welch's):")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")

# Interpretation
alpha = 0.05
if p_val < alpha:
    print("\nResult: Statistically significant difference found.")
else:
    print("\nResult: No statistically significant difference found.")

# Visualization
plt.figure(figsize=(8, 6))
plt.boxplot([both_counts, distinct_counts], labels=['Both', 'Distinct'])
plt.title('Attack Sophistication (Technique Count) by Competency Gap Type')
plt.ylabel('Number of Techniques Used')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()