import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Load the dataset
filename = 'step3_incident_coding.csv'
# Try current directory first, then parent
if os.path.exists(filename):
    file_path = filename
elif os.path.exists(f'../{filename}'):
    file_path = f'../{filename}'
else:
    raise FileNotFoundError(f"{filename} not found")

df = pd.read_csv(file_path)

# Prepare data for analysis
# Group Harm Types: Security vs Non-Security
df['harm_category'] = df['harm_type'].apply(lambda x: 'Security' if str(x).strip().lower() == 'security' else 'Non-Security')

# Group Failure Modes: Prevention vs Detection/Response
# Note: Metadata indicates 51/52 are prevention failures
df['failure_category'] = df['failure_mode'].apply(lambda x: 'Prevention' if str(x).strip().lower() == 'prevention_failure' else 'Detection/Response')

# Create Contingency Table
contingency_table = pd.crosstab(df['harm_category'], df['failure_category'])
print("=== Contingency Table (Harm Category vs Failure Mode) ===")
print(contingency_table)

# Statistical Testing
# Using Fisher's Exact Test due to expected low counts in cells
if contingency_table.shape == (2, 2):
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    print(f"\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")
else:
    print("\n[Info] Contingency table dimensions are not 2x2. One category might be missing from the data.")
    # If the table is 2x1 (e.g., only Prevention exists for one group), fill missing col with 0 for display
    if contingency_table.shape[1] == 1:
        print("All observed failures fall into a single category.")

# Visualization
# Normalize to show proportions
props = pd.crosstab(df['harm_category'], df['failure_category'], normalize='index')

fig, ax = plt.subplots(figsize=(8, 6))
props.plot(kind='bar', stacked=True, ax=ax, color=['#ff9999', '#66b3ff'])

plt.title('Proportion of Failure Modes by Harm Category')
plt.xlabel('Harm Category')
plt.ylabel('Proportion')
plt.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()