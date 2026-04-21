import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
try:
    df = pd.read_csv('../step2_competency_statements.csv')
except FileNotFoundError:
    # Fallback if running in a different environment structure, though instruction says one level above
    try:
        df = pd.read_csv('step2_competency_statements.csv')
    except FileNotFoundError:
        print("Error: Could not find step2_competency_statements.csv")
        exit(1)

# Feature Engineering: Count citations in 'evidence_summary'
# Pattern looks for strings like [#21], [#1], etc.
def count_citations(text):
    if pd.isna(text):
        return 0
    return len(re.findall(r'\[#\d+\]', str(text)))

df['citation_count'] = df['evidence_summary'].apply(count_citations)

# Descriptive Statistics
print("=== Citation Counts by Confidence Level ===")
group_stats = df.groupby('confidence')['citation_count'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(group_stats)

# Prepare data for statistical test
# We define a logical order for the groups
ordered_levels = ['High', 'Medium', 'Low']
groups = []
labels = []

for level in ordered_levels:
    if level in df['confidence'].unique():
        data = df[df['confidence'] == level]['citation_count']
        groups.append(data)
        labels.append(level)

# Check for any other labels not in High/Medium/Low
other_levels = [x for x in df['confidence'].unique() if x not in ordered_levels]
for level in other_levels:
    data = df[df['confidence'] == level]['citation_count']
    groups.append(data)
    labels.append(level)

# Statistical Test: Kruskal-Wallis H-test
# Used instead of ANOVA due to potentially small sample sizes and count data (non-normality)
print("\n=== Statistical Test Results ===")
if len(groups) > 1:
    stat, p_value = stats.kruskal(*groups)
    print(f"Test: Kruskal-Wallis H-test")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: Significant difference found (p < 0.05).")
    else:
        print("Result: No significant difference found (p >= 0.05).")
else:
    print("Insufficient groups for statistical testing.")

# Visualization
plt.figure(figsize=(8, 6))
plt.boxplot(groups, labels=labels)
plt.title('Evidence Density: Citation Counts vs. Confidence Level')
plt.ylabel('Number of Citations per Statement')
plt.xlabel('Confidence Level')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()