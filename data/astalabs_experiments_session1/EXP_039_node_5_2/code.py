import pandas as pd
import re
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define the filename
filename = 'step2_competency_statements.csv'

# robustly find the file
search_paths = [filename, f'../{filename}', f'../../{filename}']
file_path = None

for path in search_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    # Print debug info if file not found
    print(f"Could not find {filename} in searched paths: {search_paths}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Files in CWD: {os.listdir('.')}")
    try:
        print(f"Files in Parent: {os.listdir('..')}")
    except Exception as e:
        print(f"Could not list parent directory: {e}")
    raise FileNotFoundError(f"{filename} not found")

print(f"Loading dataset from {file_path}...")
df = pd.read_csv(file_path)

# Feature Engineering: Count citations in 'evidence_summary'
def count_citations(text):
    if pd.isna(text):
        return 0
    # Matches patterns like [#1], [#12], etc.
    return len(re.findall(r'\[#\d+\]', str(text)))

df['citation_count'] = df['evidence_summary'].apply(count_citations)

# Grouping by Bundle
trust_group = df[df['bundle'] == 'Trust Readiness']['citation_count']
integration_group = df[df['bundle'] == 'Integration Readiness']['citation_count']

# Descriptive Statistics
print("\n--- Descriptive Statistics for Citation Counts ---")
print(f"Trust Readiness (n={len(trust_group)}): Mean={trust_group.mean():.2f}, Std={trust_group.std():.2f}")
print(f"Integration Readiness (n={len(integration_group)}): Mean={integration_group.mean():.2f}, Std={integration_group.std():.2f}")

# Statistical Test (Welch's t-test)
t_stat, p_val = stats.ttest_ind(integration_group, trust_group, equal_var=False)
print("\n--- Welch's T-Test Results ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically significant difference found.")
else:
    print("Result: No statistically significant difference found.")

# Visualization
plt.figure(figsize=(10, 6))
# Create boxplot data
data_to_plot = [trust_group, integration_group]
labels = ['Trust Readiness', 'Integration Readiness']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

plt.title('Distribution of Literature Citations per Competency Bundle')
plt.ylabel('Citation Count per Statement')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()