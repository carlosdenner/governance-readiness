import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

# Define file path handling
filename = 'step2_crosswalk_matrix.csv'
# Try checking parent directory first as per instructions
filepath = os.path.join('..', filename)
if not os.path.exists(filepath):
    filepath = filename  # Fallback to current dir

print(f"Loading dataset from: {filepath}")

try:
    df = pd.read_csv(filepath)
except FileNotFoundError:
    print(f"Error: File {filename} not found in current or parent directory.")
    sys.exit(1)

# --- Data Preprocessing ---

# Function to categorize frameworks
def categorize_framework(source_text):
    if pd.isna(source_text):
        return None
    source_upper = source_text.upper()
    if 'EU AI ACT' in source_upper:
        return 'EU'
    elif 'NIST' in source_upper:
        return 'NIST'
    else:
        return 'Other'

# Apply categorization
df['Framework_Family'] = df['source'].apply(categorize_framework)

# Filter for analysis (Exclude 'Other')
df_analysis = df[df['Framework_Family'] != 'Other'].copy()

print("\n=== Data Summary ===")
print(f"Total records loaded: {len(df)}")
print(f"Records in analysis (EU vs NIST): {len(df_analysis)}")
print(df_analysis['Framework_Family'].value_counts())

# --- Contingency Table ---
# We want to see: Framework vs Bundle
# Orient columns for Fisher's Exact Test logic: 
# Target comparison: Trust Readiness vs Integration Readiness
contingency = pd.crosstab(df_analysis['Framework_Family'], df_analysis['bundle'])

# Reorder specifically for Odds Ratio calculation:
# Rows: EU, NIST
# Cols: Trust Readiness, Integration Readiness
desired_index = ['EU', 'NIST']
desired_columns = ['Trust Readiness', 'Integration Readiness']

# Ensure all keys exist, fill with 0 if missing
contingency_ordered = contingency.reindex(index=desired_index, columns=desired_columns, fill_value=0)

print("\n=== Contingency Table (Observed Counts) ===")
print(contingency_ordered)

# --- Statistical Analysis ---

# 1. Chi-Square Test of Independence
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_ordered)

print("\n=== Statistical Test Results ===")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"Chi-Square P-value:   {p_chi2:.4f}")

# 2. Fisher's Exact Test (More appropriate for small sample sizes)
odds_ratio, p_fisher = stats.fisher_exact(contingency_ordered)

print(f"Fisher's Exact P-value: {p_fisher:.4f}")
print(f"Odds Ratio:             {odds_ratio:.4f}")

# Interpretation of Odds Ratio
# OR = (Odds of Trust given EU) / (Odds of Trust given NIST)
print("\n--- Interpretation ---")
if odds_ratio > 1:
    print(f"The odds of a requirement mapping to 'Trust Readiness' are {odds_ratio:.2f} times higher for EU AI Act than for NIST.")
elif odds_ratio < 1:
    print(f"The odds of a requirement mapping to 'Trust Readiness' are {1/odds_ratio:.2f} times higher for NIST than for EU AI Act.")
else:
    print("No difference in odds between frameworks.")

# --- Visualization ---

# Calculate proportions for plotting
props = contingency_ordered.div(contingency_ordered.sum(axis=1), axis=0)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plot stacked bar chart
# Use specific colors for clarity: Trust (Greenish), Integration (Blueish)
props.plot(kind='bar', stacked=True, ax=ax, color=['#2ca02c', '#1f77b4'], alpha=0.8)

plt.title('Proportion of Competency Bundles by Framework Family')
plt.xlabel('Framework Family')
plt.ylabel('Proportion')
plt.legend(title='Competency Bundle', loc='upper right', bbox_to_anchor=(1.25, 1))
plt.xticks(rotation=0)

# Annotate bars with counts
for n, x in enumerate([*contingency_ordered.index.values]):
    for (cn, y) in enumerate(props.loc[x]):
        if y > 0:
            # Calculate cumulative height for position
            y_pos = props.loc[x].iloc[:cn].sum() + y/2
            # Get raw count
            raw_count = contingency_ordered.loc[x].iloc[cn]
            plt.text(n, y_pos, f"{raw_count}\n({y:.0%})", 
                     ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
