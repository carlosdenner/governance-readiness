import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Try to locate the file in the current directory or one level up
filename = 'step2_crosswalk_matrix.csv'
if os.path.exists(filename):
    file_path = filename
elif os.path.exists(f'../{filename}'):
    file_path = f'../{filename}'
else:
    raise FileNotFoundError(f"Could not find {filename} in current or parent directory.")

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Target columns
control_col = 'Audit Logging & Telemetry'
bundle_col = 'bundle'

# Clean and prepare data
# Treat 'X' as Present (1) and NaN/Empty as Absent (0)
df['has_control'] = df[control_col].fillna('').apply(lambda x: 1 if str(x).strip().upper() == 'X' else 0)

# Create Contingency Table
contingency_table = pd.crosstab(df[bundle_col], df['has_control'])

# Ensure both 0 (Absent) and 1 (Present) columns exist
if 0 not in contingency_table.columns:
    contingency_table[0] = 0
if 1 not in contingency_table.columns:
    contingency_table[1] = 0

# Reorder columns to Absent (0), Present (1)
contingency_table = contingency_table[[0, 1]]
contingency_table.columns = ['Absent', 'Present']

print("\n--- Contingency Table (Count) ---")
print(contingency_table)

# Calculate Percentages
contingency_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
print("\n--- Contingency Table (Percentage) ---")
print(contingency_pct)

# Perform Fisher's Exact Test
try:
    # Extract counts for the test
    # Row 1: Integration Readiness
    # Row 2: Trust Readiness
    # Columns: Present, Absent (swapped from table for Odds Ratio interpretation: Present/Absent)
    
    ir_present = contingency_table.loc['Integration Readiness', 'Present']
    ir_absent = contingency_table.loc['Integration Readiness', 'Absent']
    tr_present = contingency_table.loc['Trust Readiness', 'Present']
    tr_absent = contingency_table.loc['Trust Readiness', 'Absent']
    
    # Matrix for Fisher's: [[Trust_Present, Trust_Absent], [Integration_Present, Integration_Absent]]
    # Testing if Trust has higher prevalence than Integration
    fisher_matrix = [[tr_present, tr_absent], [ir_present, ir_absent]]
    
    odds_ratio, p_value = stats.fisher_exact(fisher_matrix, alternative='two-sided')

    print("\n--- Statistical Test Results ---")
    print(f"Fisher's Exact Test P-value: {p_value:.4f}")
    print(f"Odds Ratio (Trust/Integration): {odds_ratio:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print("Result: Statistically Significant (Reject Null Hypothesis)")
    else:
        print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")

except KeyError as e:
    print(f"Error processing bundles for stats: {e}")

# Visualization
plt.figure(figsize=(8, 6))
bundles = contingency_pct.index
percentages = contingency_pct['Present']

# Color mapping
colors = ['salmon' if 'Trust' in b else 'skyblue' for b in bundles]

bars = plt.bar(bundles, percentages, color=colors)

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
