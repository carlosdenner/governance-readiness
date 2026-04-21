import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset
file_path = '../step2_competency_statements.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
except FileNotFoundError:
    # Fallback if running in a different environment structure, though instruction said one level up
    try:
        df = pd.read_csv('step2_competency_statements.csv')
        print("Loaded from current directory")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Calculate word count for each competency statement
# Using simple whitespace splitting for word count
df['word_count'] = df['competency_statement'].fillna('').astype(str).apply(lambda x: len(x.split()))

# Group data by bundle
integration_group = df[df['bundle'] == 'Integration Readiness']['word_count']
trust_group = df[df['bundle'] == 'Trust Readiness']['word_count']

# Calculate descriptive statistics
stats_summary = df.groupby('bundle')['word_count'].describe()
print("\n=== Descriptive Statistics (Word Count) ===")
print(stats_summary)

# Perform Independent T-test (Welch's t-test, assuming unequal variances)
t_stat, p_val = stats.ttest_ind(integration_group, trust_group, equal_var=False)

print("\n=== Statistical Test Results ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")
if p_val < 0.05:
    print("Result: Significant difference in word counts between bundles.")
else:
    print("Result: No significant difference in word counts between bundles.")

# Visualization
plt.figure(figsize=(10, 6))
plt.boxplot([integration_group, trust_group], labels=['Integration Readiness', 'Trust Readiness'])
plt.title('Distribution of Competency Statement Word Counts by Bundle')
plt.ylabel('Word Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()