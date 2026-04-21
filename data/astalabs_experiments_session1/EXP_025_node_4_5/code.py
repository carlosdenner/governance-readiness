# [debug]
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import re
import sys

# Define file path (one level up as per instructions)
file_path = '../step4_propositions.csv'

print("=== Loading Dataset ===")
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

print("\n=== Processing Evidence Counts ===")
# Function to count atlas evidence references
def count_evidence(evidence_str):
    if pd.isna(evidence_str) or evidence_str.strip() == '':
        return 0
    # Strategy 1: Count occurrences of AML.CS patterns (case study IDs)
    matches = re.findall(r'AML\.CS\d+', str(evidence_str))
    if matches:
        return len(set(matches)) # Unique case studies
    
    # Strategy 2: If no IDs found, split by semicolon as fallback for list formats
    return len([x for x in str(evidence_str).split(';') if x.strip()])

df['evidence_count'] = df['atlas_evidence'].apply(count_evidence)

# Display the extracted counts for verification
print(df[['proposition_id', 'confidence', 'atlas_evidence', 'evidence_count']])

print("\n=== Grouping by Confidence ===")
# Check distribution of confidence levels
confidence_counts = df['confidence'].value_counts()
print("Confidence level distribution:")
print(confidence_counts)

# Create groups
high_conf = df[df['confidence'].str.lower() == 'high']['evidence_count']
other_conf = df[df['confidence'].str.lower().isin(['medium', 'low'])]['evidence_count']

stats_summary = df.groupby('confidence')['evidence_count'].describe()
print("\nDescriptive Statistics by Confidence:")
print(stats_summary)

print("\n=== Statistical Testing ===")
# We check if we have enough data for a test. 
# With only 5 propositions, this is illustrative.
if len(high_conf) > 0 and len(other_conf) > 0:
    # T-test (High vs Medium/Low)
    t_stat, p_val = stats.ttest_ind(high_conf, other_conf, equal_var=False)
    print(f"T-test (High vs Medium/Low): t={t_stat:.4f}, p={p_val:.4f}")
else:
    print("Insufficient data groups for statistical testing.")

print("\n=== Plotting ===")
# Bar chart of evidence counts
plt.figure(figsize=(10, 6))
# Calculate means for plotting
means = df.groupby('confidence')['evidence_count'].mean()
# Reorder if indices allow (High, Medium, Low)
order = [x for x in ['High', 'Medium', 'Low'] if x in means.index]
means = means.reindex(order)

colors = ['#2ca02c' if c == 'High' else '#ff7f0e' if c == 'Medium' else '#d62728' for c in means.index]

plt.bar(means.index, means.values, color=colors)
plt.title('Average Atlas Evidence Count by Confidence Level')
plt.xlabel('Confidence Assessment')
plt.ylabel('Avg. Number of Linked Case Studies')
plt.grid(axis='y', alpha=0.3)

# Add individual data points since N is small
for conf in means.index:
    subset = df[df['confidence'] == conf]['evidence_count']
    x_vals = [conf] * len(subset)
    plt.scatter(x_vals, subset, color='black', zorder=5, alpha=0.7, label='Individual Props' if conf == means.index[0] else "")

if 'Individual Props' in plt.gca().get_legend_handles_labels()[1]:
    plt.legend()

plt.show()
