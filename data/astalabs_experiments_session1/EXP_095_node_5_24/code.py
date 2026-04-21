import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
# Attempting to load from current directory as '../' failed in previous step
try:
    df = pd.read_csv('step3_incident_coding.csv')
except FileNotFoundError:
    # Fallback to absolute path or check if it's in a subdirectory if needed, 
    # but strictly following previous success patterns implies current dir.
    print("File not found in current directory. Creating dummy data for structure verification if needed, or exiting.")
    raise

# 1. Date Parsing
df['incident_date_dt'] = pd.to_datetime(df['incident_date'], errors='coerce')

# Filter out rows with invalid dates if any
df = df.dropna(subset=['incident_date_dt'])

# 2. Define Eras
cutoff_date = pd.Timestamp('2023-01-01')
df['era'] = df['incident_date_dt'].apply(lambda d: 'Post-2023' if d >= cutoff_date else 'Pre-2023')

# 3. Categorize Harm Types
# Prompt: 'Societal' (bias_discrimination, privacy)
# Prompt: 'Security/Reliability' (security, reliability, supply_chain)
societal_harms = ['bias_discrimination', 'privacy']
security_harms = ['security', 'reliability', 'supply_chain']

def categorize_harm(h_type):
    h_type = str(h_type).strip()
    if h_type in societal_harms:
        return 'Societal'
    elif h_type in security_harms:
        return 'Security/Reliability'
    else:
        return 'Other'

df['harm_category'] = df['harm_type'].apply(categorize_harm)

# Filter out 'Other' to test the specific hypothesis strictly
df_filtered = df[df['harm_category'] != 'Other'].copy()

print("=== Data Summary ===")
print(f"Total incidents processed: {len(df)}")
print(f"Incidents in hypothesis categories: {len(df_filtered)}")
print("\nHarm Category Counts:")
print(df_filtered['harm_category'].value_counts())
print("\nEra Counts:")
print(df_filtered['era'].value_counts())

# 4. Contingency Table
contingency_table = pd.crosstab(df_filtered['era'], df_filtered['harm_category'])
print("\n=== Contingency Table ===")
print(contingency_table)

# 5. Statistical Test (Chi-Square)
# Note: If counts are low (<5 in cells), Fisher's Exact Test is preferred, but for 2x2 Chi2 is standard start.
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n=== Statistical Test Results ===")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")

# Fisher's Exact Test (since it's 2x2 and likely small sample size)
if contingency_table.shape == (2, 2):
    # fisher_exact returns (odds_ratio, p_value)
    odds_ratio, fisher_p = stats.fisher_exact(contingency_table)
    print(f"Fisher's Exact Test P-value: {fisher_p:.4f}")
    print(f"Odds Ratio: {odds_ratio:.4f}")

# 6. Visualization
# Calculate proportions for plotting
props = contingency_table.div(contingency_table.sum(axis=1), axis=0)

ax = props.plot(kind='bar', stacked=True, color=['#d9534f', '#5bc0de'], figsize=(10, 6))
plt.title('Proportion of Harm Categories by Era (Pre vs Post 2023)')
plt.ylabel('Proportion')
plt.xlabel('Era')
plt.legend(title='Harm Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Add count annotations
for n, x in enumerate([*contingency_table.index.values]):
    for (proportion, count, y_loc) in zip(props.loc[x], contingency_table.loc[x], props.loc[x].cumsum()):                
        plt.text(x=n, y=(y_loc - proportion) + (proportion / 2), s=f'{count} ({proportion:.1%})', 
                 color="white", fontsize=10, fontweight="bold", ha="center", va="center")

plt.show()