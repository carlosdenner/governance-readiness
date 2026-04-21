import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import sys

# [debug]
print("Starting experiment...")

# Robust file loading
filename = 'step2_crosswalk_matrix.csv'
filepath = None

if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    print(f"Error: File {filename} not found in current or parent directory.")
    print(f"Current working directory: {os.getcwd()}")
    try:
        print(f"Listing current dir: {os.listdir('.')}")
        print(f"Listing parent dir: {os.listdir('..')}")
    except Exception as e:
        print(f"Could not list directories: {e}")
    sys.exit(1)

print(f"Loading dataset from: {filepath}")
df = pd.read_csv(filepath)

# Clean column names
df.columns = [c.strip() for c in df.columns]

# Verify columns
rag_col = 'RAG Architecture & Data Grounding'
bundle_col = 'bundle'

if rag_col not in df.columns or bundle_col not in df.columns:
    print(f"Required columns not found. Available: {df.columns.tolist()}")
    sys.exit(1)

# Filter for relevant bundles (just in case there are others/nans)
target_bundles = ['Integration Readiness', 'Trust Readiness']
df = df[df[bundle_col].isin(target_bundles)].copy()

# Create binary variable for RAG control
# Assuming 'X' marks presence, anything else is absence
df['has_rag'] = df[rag_col].apply(lambda x: 1 if str(x).strip().upper() == 'X' else 0)

# Create Contingency Table
# Rows: Bundle, Columns: Has RAG (0, 1)
contingency = pd.crosstab(df[bundle_col], df['has_rag'])

# Ensure columns 0 and 1 exist
for c in [0, 1]:
    if c not in contingency.columns:
        contingency[c] = 0
contingency = contingency[[0, 1]]

print("\n=== Contingency Table (Count) ===")
print(contingency)

# Calculate percentages for reporting
contingency_pct = pd.crosstab(df[bundle_col], df['has_rag'], normalize='index') * 100
print("\n=== Contingency Table (Percentage) ===")
print(contingency_pct)

# Fisher's Exact Test
# We want to test if Integration Readiness is MORE likely to have RAG than Trust Readiness.
# Construct 2x2 matrix: [[Integration_Yes, Integration_No], [Trust_Yes, Trust_No]]

# Get counts
try:
    int_yes = contingency.loc['Integration Readiness', 1]
    int_no = contingency.loc['Integration Readiness', 0]
    trust_yes = contingency.loc['Trust Readiness', 1]
    trust_no = contingency.loc['Trust Readiness', 0]
except KeyError as e:
    print(f"Error accessing bundle keys: {e}")
    sys.exit(1)

# Table for stats: [[Yes, No]] for Group 1, then Group 2
# This aligns with Odds Ratio = (Yes1/No1) / (Yes2/No2)
stats_table = [
    [int_yes, int_no],      # Integration Readiness
    [trust_yes, trust_no]   # Trust Readiness
]

# Perform Fisher's Exact Test (Two-sided to be conservative, check OR for direction)
odds_ratio, p_value = stats.fisher_exact(stats_table, alternative='two-sided')

print("\n=== Fisher's Exact Test Results ===")
print(f"Comparison: Integration Readiness vs. Trust Readiness")
print(f"Integration RAG Rate: {int_yes}/{int_yes+int_no} ({int_yes/(int_yes+int_no)*100:.1f}%)")
print(f"Trust RAG Rate:       {trust_yes}/{trust_yes+trust_no} ({trust_yes/(trust_yes+trust_no)*100:.1f}%)")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value:    {p_value:.4f}")

interpretation = ""
if p_value < 0.05:
    interpretation = "Statistically Significant Difference."
else:
    interpretation = "No Statistically Significant Difference."

if odds_ratio > 1:
    direction = "Integration Readiness is more associated with RAG."
elif odds_ratio < 1:
    direction = "Trust Readiness is more associated with RAG."
else:
    direction = "No directional difference."

print(f"Conclusion: {interpretation} {direction}")

# Visualization
plt.figure(figsize=(8, 6))
# Plot percentage of 'Yes' (column 1)
ax = contingency_pct[1].plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black', rot=0)

plt.title('Prevalence of "RAG Architecture & Data Grounding" Control')
plt.xlabel('Competency Bundle')
plt.ylabel('Percentage of Requirements (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add count labels
for i, p in enumerate(ax.patches):
    bundle_name = contingency_pct.index[i]
    count = contingency.loc[bundle_name, 1]
    total = contingency.loc[bundle_name].sum()
    height = p.get_height()
    ax.annotate(f'{height:.1f}%\n(n={count}/{total})',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()