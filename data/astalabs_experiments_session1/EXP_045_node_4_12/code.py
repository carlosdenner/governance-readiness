import json
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# [debug]
# print("Starting experiment...")

# 1. Load the dataset
file_path = 'step3_enrichments.json'
try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    # Try looking one level up as per instructions if not found, 
    # but instructions say "Dataset files are present one level above" 
    # usually implies we might need ../ but previous context suggests 
    # just filename works or I should handle the path.
    # However, standard instruction says "Use only the dataset given".
    # I will assume it is in the current dir or handle the error if strict.
    # Actually, the prompt says "Dataset files are present one level above the current working directory."
    # So I should use ../step3_enrichments.json
    with open('../' + file_path, 'r') as f:
        data = json.load(f)
except Exception as e:
    # Fallback to current directory if ../ fails (just in case)
    with open(file_path, 'r') as f:
        data = json.load(f)

# 2. Extract relevant fields
records = []
for entry in data:
    split = entry.get('trust_integration_split', '').lower()
    cot = entry.get('chain_of_thought', '')
    
    # Define group
    if 'both' in split:
        group = 'Both'
    elif 'trust' in split or 'integration' in split:
        group = 'Distinct'
    else:
        group = 'Other' # Should not happen based on metadata, but safety check
        
    records.append({
        'case_study_id': entry.get('case_study_id'),
        'trust_integration_split': split,
        'group': group,
        'cot_length': len(cot)
    })

df = pd.DataFrame(records)

# Filter out 'Other' if any
df = df[df['group'] != 'Other']

# 3. Summary Statistics
group_stats = df.groupby('group')['cot_length'].agg(['count', 'mean', 'std', 'min', 'max'])
print("=== Chain-of-Thought Length Statistics by Group ===")
print(group_stats)
print("\n")

# 4. Statistical Test (Independent Samples T-Test)
group_both = df[df['group'] == 'Both']['cot_length']
group_distinct = df[df['group'] == 'Distinct']['cot_length']

print(f"Group 'Both' n={len(group_both)}")
print(f"Group 'Distinct' n={len(group_distinct)}")

# Check assumptions: variances (Levene's test) - optional but good practice
stat_lev, p_lev = stats.levene(group_both, group_distinct)
print(f"Levene's test for equal variances: p={p_lev:.4f}")

# Perform T-test (Welch's t-test recommended if sample sizes or variances differ)
t_stat, p_val = stats.ttest_ind(group_both, group_distinct, equal_var=(p_lev > 0.05))

print("=== T-Test Results ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically significant difference in chain-of-thought length.")
else:
    print("Result: No statistically significant difference detected.")

# 5. Visualization
plt.figure(figsize=(8, 6))
# Create a boxplot
data_to_plot = [group_both, group_distinct]
plt.boxplot(data_to_plot, labels=['Both (Trust & Integration)', 'Distinct (Single Domain)'])
plt.title('Distribution of Chain-of-Thought Length by Complexity')
plt.ylabel('Character Count')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()