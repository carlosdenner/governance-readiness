import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# [debug]
print("Current working directory:", os.getcwd())

# Define file path based on instructions
filename = 'step2_crosswalk_matrix.csv'
filepath = f"../{filename}"

if not os.path.exists(filepath):
    # Fallback to current directory if not found in parent
    filepath = filename

print(f"Loading dataset from: {filepath}")

try:
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File {filename} not found.")
    exit(1)

# Target Columns
bundle_col = 'bundle'
control_col = 'Human-in-the-Loop Approval Gates'

# Check if columns exist
if control_col not in df.columns:
    print(f"Column '{control_col}' not found. Available columns:")
    print(df.columns.tolist())
    exit(1)

# Preprocess: Convert control column to boolean (True if 'X' or non-null/non-empty, False otherwise)
# Looking at previous exploration, 'X' indicates presence.
df['has_hitl'] = df[control_col].notna() & (df[control_col].astype(str).str.strip() != '')

# Create Contingency Table
contingency_table = pd.crosstab(df[bundle_col], df['has_hitl'])
contingency_table.columns = ['No HITL', 'Has HITL']

print("\n--- Contingency Table ---")
print(contingency_table)

# Perform Fisher's Exact Test
# Fisher's exact test requires a 2x2 table. 
# Ensure we have both bundles and both presence/absence states if possible, though crosstab handles available data.
if contingency_table.shape == (2, 2):
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    print(f"\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print("Result: Statistically Significant association between Bundle and HITL Control.")
    else:
        print("Result: No statistically significant association found.")
else:
    print("\nContingency table is not 2x2. Cannot perform Fisher's Exact Test.")
    print("Shape:", contingency_table.shape)

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Association: Bundle vs Human-in-the-Loop Control')
plt.ylabel('Competency Bundle')
plt.xlabel('Has Human-in-the-Loop Approval Gates?')
plt.tight_layout()
plt.show()