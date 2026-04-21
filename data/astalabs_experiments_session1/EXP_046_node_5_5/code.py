import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import sys

# Robust file loading
filename = 'step2_crosswalk_matrix.csv'
file_path = None

# Check current directory and parent directory
possible_paths = [filename, f'../{filename}']

for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    print(f"Error: {filename} not found in current or parent directory.")
    sys.exit(1)

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Target column and bundle column
control_col = 'Human-in-the-Loop Approval Gates'
bundle_col = 'bundle'

# Verify columns exist
if control_col not in df.columns or bundle_col not in df.columns:
    print(f"Error: Required columns not found. Available columns: {df.columns.tolist()}")
    sys.exit(1)

# Preprocess: Convert control column to binary (1 if 'X', 0 otherwise)
# Fill NA with empty string, convert to string, strip whitespace, check for 'X'
df['is_mapped'] = df[control_col].fillna('').astype(str).str.strip().apply(lambda x: 1 if x.upper() == 'X' else 0)

# Group by bundle to see raw counts and proportions
summary = df.groupby(bundle_col)['is_mapped'].agg(['count', 'sum', 'mean'])
summary.rename(columns={'count': 'Total Requirements', 'sum': 'Control Mapped Count', 'mean': 'Usage Proportion'}, inplace=True)
print("\n=== Descriptive Statistics ===")
print(summary)

# Create Contingency Table for Statistical Test
# Rows: Bundle (Trust vs Integration)
# Cols: Control Mapped (0 vs 1)
contingency_table = pd.crosstab(df[bundle_col], df['is_mapped'])
print("\n=== Contingency Table ===")
print(contingency_table)

# Check if we have enough data for a valid test (at least 2x2)
if contingency_table.shape == (2, 2):
    # Perform Chi-square test of independence
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    print("\n=== Statistical Test Results (Chi-square) ===")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4f}")

    # Fisher's Exact Test (often better for small sample sizes)
    odds_ratio, fisher_p = stats.fisher_exact(contingency_table)
    print(f"Fisher's Exact Test p-value: {fisher_p:.4f}")
    print(f"Odds Ratio: {odds_ratio:.4f}")
else:
    print("\nWarning: Contingency table is not 2x2. One bundle might have 0 mappings.")
    # If one bundle has 0 mappings, we can still do a Fisher exact test if we construct the 2x2 manually or handle it
    # But let's see the output first. 

# Visualization
plt.figure(figsize=(8, 6))
# We want to plot the proportion of 'is_mapped' == 1 for each bundle
proportions = summary['Usage Proportion']
ax = proportions.plot(kind='bar', color=['skyblue', 'salmon'], alpha=0.8)
plt.title(f"Proportion of Requirements Mapping to\n'{control_col}' by Bundle")
plt.ylabel("Proportion (Usage Rate)")
plt.xlabel("Competency Bundle")
plt.ylim(0, 1.0)

# Add value labels
for i, v in enumerate(proportions):
    ax.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
