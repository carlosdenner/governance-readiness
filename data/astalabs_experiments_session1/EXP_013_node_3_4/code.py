import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Load dataset
try:
    df = pd.read_csv('../step2_crosswalk_matrix.csv')
except FileNotFoundError:
    try:
        df = pd.read_csv('step2_crosswalk_matrix.csv')
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Define target column and bundle column
control_col = 'Human-in-the-Loop Approval Gates'
bundle_col = 'bundle'

# Preprocess: Convert 'X' to 1, others to 0
df['has_control'] = df[control_col].apply(lambda x: 1 if str(x).strip().upper() == 'X' else 0)

# Generate Contingency Table
# We ensure all possible values (0 and 1) are present for both bundles to ensure a 2x2 matrix
contingency = pd.crosstab(df[bundle_col], df['has_control'])

# Ensure columns 0 and 1 exist (in case one is missing entirely from the dataset)
if 0 not in contingency.columns:
    contingency[0] = 0
if 1 not in contingency.columns:
    contingency[1] = 0
    
# Sort columns to be [0, 1] for consistent interpretation: 0=Absent, 1=Present
contingency = contingency[[0, 1]]

print("=== Contingency Table (Rows=Bundle, Cols=Control Presence) ===")
print(contingency)
print("\n")

# Calculate percentages for reporting
bundle_stats = df.groupby(bundle_col)['has_control'].agg(['count', 'sum', 'mean'])
bundle_stats['percentage'] = bundle_stats['mean'] * 100
print("=== Descriptive Statistics ===")
print(bundle_stats)
print("\n")

# Perform Fisher's Exact Test
# Fisher's is chosen over Chi-square due to small sample size (N=42)
# The contingency table index usually sorts alphabetically: Integration, Trust.
# We need to pass the 2x2 matrix.
try:
    odds_ratio, p_value = stats.fisher_exact(contingency)
    print("=== Fisher's Exact Test Results ===")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: Statistically Significant (p < {alpha})")
    else:
        print(f"Result: Not Statistically Significant (p >= {alpha})")
except Exception as e:
    print(f"Error performing statistical test: {e}")

# Visualization
plt.figure(figsize=(8, 6))
colors = ['skyblue' if b == 'Integration Readiness' else 'salmon' for b in bundle_stats.index]
bars = plt.bar(bundle_stats.index, bundle_stats['percentage'], color=colors, edgecolor='black')

plt.title(f"Prevalence of '{control_col}' by Bundle")
plt.ylabel('Percentage of Requirements with Control (%)')
plt.xlabel('Competency Bundle')
plt.ylim(0, 100)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()