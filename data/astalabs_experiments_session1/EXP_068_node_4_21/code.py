import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# [debug] Check file existence
file_name = 'step3_incident_coding.csv'
possible_paths = [f'../{file_name}', file_name]
file_path = None
for p in possible_paths:
    if os.path.exists(p):
        file_path = p
        break

if not file_path:
    print(f"File {file_name} not found in checked paths: {possible_paths}")
    # Attempt to use step3_enrichments.json as fallback if csv is missing, as they share structure in metadata descriptions
    json_file = 'step3_enrichments.json'
    possible_json_paths = [f'../{json_file}', json_file]
    for p in possible_json_paths:
        if os.path.exists(p):
            print(f"Falling back to {p}")
            file_path = p
            break

if not file_path:
    raise FileNotFoundError("Neither incident coding CSV nor enrichments JSON found.")

# Load Data
print(f"Loading dataset from {file_path}...")
if file_path.endswith('.csv'):
    df = pd.read_csv(file_path)
else:
    df = pd.read_json(file_path)

# Verify columns
required_cols = ['harm_type', 'failure_mode']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols}. Available: {df.columns.tolist()}")
    exit(1)

# Preprocessing
# Clean whitespace and standardize case
df['harm_type'] = df['harm_type'].fillna('').astype(str).str.strip().str.lower()
df['failure_mode'] = df['failure_mode'].fillna('').astype(str).str.strip().str.lower()

# Create Binary Harm Category
df['harm_category'] = df['harm_type'].apply(lambda x: 'Security' if 'security' in x else 'Other')

# Generate Contingency Table
contingency = pd.crosstab(df['harm_category'], df['failure_mode'])
print("\n--- Contingency Table (Observed) ---")
print(contingency)

# Check for empty columns or rows which might break Chi2
if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
    print("\nWarning: Contingency table too small for Chi-Square test (need at least 2x2).")
else:
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n--- Chi-Square Test Results ---")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print("\nExpected Frequencies:")
    print(expected)

# Visualization
# Calculate proportions for stacked bar chart
contingency_prop = contingency.div(contingency.sum(axis=1), axis=0)

# Plot
plt.figure(figsize=(10, 6))
# contingency_prop.plot(kind='bar', stacked=True) is cleaner, but let's use explicit ax for control
ax = contingency_prop.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))

plt.title('Proportion of Failure Modes by Harm Category')
plt.xlabel('Harm Category')
plt.ylabel('Proportion')
plt.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()

print("\nDisplaying plot...")
plt.show()
