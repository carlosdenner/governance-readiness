import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

# Define file path (one level above as per instructions)
dataset_path = '../step3_incident_coding.csv'

# Check if file exists, fallback to current dir if not (for robustness)
if not os.path.exists(dataset_path):
    dataset_path = 'step3_incident_coding.csv'

try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"Error: Could not find dataset at {dataset_path}")
    sys.exit(1)

print("=== Loading and Preprocessing Data ===")
print(f"Dataset loaded: {dataset_path}")
print(f"Shape: {df.shape}")

# --- Step 1: Feature Engineering (Prompt Injection) ---
# We check 'techniques_used' for the string 'Prompt Injection'. 
# We also check for 'AML.T0051' (ATLAS ID for LLM Prompt Injection) just in case the column uses IDs.
df['techniques_used'] = df['techniques_used'].fillna('')
df['has_prompt_injection'] = df['techniques_used'].astype(str).str.contains('Prompt Injection|AML.T0051', case=False, regex=True)

print("\n--- Distribution of Prompt Injection ---")
print(df['has_prompt_injection'].value_counts())

# Debug: Show a few techniques to confirm format
print("\n[Debug] First 5 'techniques_used' entries:")
print(df['techniques_used'].head().tolist())

# --- Step 2: Feature Engineering (Failure Mode) ---
# Categorize into Prevention vs. Non-Prevention (Detection/Response)
def categorize_failure(mode):
    if pd.isna(mode):
        return 'Unknown'
    mode_str = str(mode).lower()
    if 'prevention' in mode_str:
        return 'Prevention'
    elif 'detection' in mode_str or 'response' in mode_str:
        return 'Detection/Response'
    else:
        return 'Other'

df['failure_category'] = df['failure_mode'].apply(categorize_failure)

print("\n--- Failure Category Distribution ---")
print(df['failure_category'].value_counts())

# --- Step 3: Contingency Table ---
# Rows: Has Prompt Injection (False/True)
# Cols: Failure Category (Prevention/Detection+Response)
contingency_table = pd.crosstab(df['has_prompt_injection'], df['failure_category'])

# Ensure we have the specific columns we want to test
expected_cols = ['Prevention', 'Detection/Response']
for col in expected_cols:
    if col not in contingency_table.columns:
        contingency_table[col] = 0

# Reorder for consistency
contingency_table = contingency_table[expected_cols]

print("\n--- Contingency Table (Observed) ---")
print(contingency_table)

# --- Step 4: Statistical Test ---
# Using Fisher's Exact Test due to potential small sample sizes in the 'Detection/Response' column
try:
    # Fisher's Exact Test requires a 2x2 table
    # Table structure: [[No_Prev, No_Det], [Yes_Prev, Yes_Det]]
    if contingency_table.shape == (2, 2):
        odds_ratio, p_value = stats.fisher_exact(contingency_table)
        print("\n=== Statistical Test Results (Fisher's Exact Test) ===")
        print(f"Odds Ratio: {odds_ratio}")
        print(f"P-value: {p_value:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            print("Conclusion: Statistically SIGNIFICANT association between Prompt Injection and Failure Mode.")
        else:
            print("Conclusion: NO statistically significant association found.")
    else:
        print("\nCannot perform Fisher's Exact Test: Contingency table is not 2x2 (likely missing one category entirely).")
except Exception as e:
    print(f"\nError performing statistical test: {e}")

# --- Step 5: Visualization ---
# Grouped bar chart to visualize the counts
# We plot the contingency table directly
ax = contingency_table.plot(kind='bar', figsize=(10, 6), rot=0, color=['#1f77b4', '#ff7f0e'])

plt.title('Failure Mode Distribution by Presence of Prompt Injection')
plt.xlabel('Has Prompt Injection')
plt.ylabel('Count of Incidents')
plt.xticks([0, 1], ['No', 'Yes'])
plt.legend(title='Failure Category')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()