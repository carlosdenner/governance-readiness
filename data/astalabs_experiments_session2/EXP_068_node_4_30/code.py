import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

# [debug]
print("Starting experiment: Realized Harm Bias Analysis...")

# 1. Load the dataset
filename = 'astalabs_discovery_all_data.csv'
file_path = filename
if not os.path.exists(file_path):
    file_path = os.path.join('..', filename)
    if not os.path.exists(file_path):
        print(f"Error: Dataset {filename} not found.")
        sys.exit(1)

try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# 2. Filter for AIID incidents
df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(df_aiid)} rows")

# 3. Categorize 'Harm Distribution Basis' (Protected vs General)
col_basis = 'Harm Distribution Basis'

def classify_basis(basis):
    basis_str = str(basis).lower()
    if basis_str == 'nan' or basis_str == 'none' or basis_str == '':
        return 'General Public/Other'
        
    protected_keywords = [
        'race', 'racial', 'gender', 'sex', 'woman', 'man', 'black', 'white', 
        'asian', 'latino', 'hispanic', 'indigenous', 'native', 'ethnic', 'ethnicity',
        'religion', 'religious', 'muslim', 'jewish', 'christian', 'hindu', 
        'age', 'elderly', 'senior', 'child', 'minor', 'youth', 
        'disability', 'disabled', 'handicap', 
        'sexual orientation', 'lgbt', 'queer', 'gay', 'lesbian', 'transgender'
    ]
    if any(k in basis_str for k in protected_keywords):
        return 'Protected Class'
    return 'General Public/Other'

df_aiid['Target_Group_Type'] = df_aiid[col_basis].apply(classify_basis)

print("\n--- Target Group Distribution ---")
print(df_aiid['Target_Group_Type'].value_counts())

# 4. Categorize 'AI Harm Level' (Realized vs Potential)
# Based on previous exploration, values are: 
# 'AI tangible harm event', 'AI tangible harm near-miss', 'AI tangible harm issue', 'unclear', 'none'

col_level = 'AI Harm Level'

def classify_harm_status(val):
    s = str(val).lower().strip()
    if 'event' in s:
        return 'Realized Harm' # The incident actually happened and caused harm
    elif 'near-miss' in s or 'issue' in s:
        return 'Potential Harm' # Near miss or identified issue without realized harm
    else:
        return np.nan # Exclude unclear/none

df_aiid['Harm_Status'] = df_aiid[col_level].apply(classify_harm_status)

# Filter for valid harm status
df_final = df_aiid.dropna(subset=['Harm_Status'])
print(f"\nRows with valid Harm Status: {len(df_final)}")
print("--- Harm Status Distribution ---")
print(df_final['Harm_Status'].value_counts())

# 5. Statistical Analysis (Chi-Square Test)
# Create contingency table
contingency_table = pd.crosstab(df_final['Target_Group_Type'], df_final['Harm_Status'])
print("\n--- Contingency Table (Counts) ---")
print(contingency_table)

# Calculate percentages for better interpretation
props = pd.crosstab(df_final['Target_Group_Type'], df_final['Harm_Status'], normalize='index') * 100
print("\n--- Contingency Table (Percentages) ---")
print(props.round(2))

# Check if we have enough data for Chi-Square
if contingency_table.size == 4 and (contingency_table > 5).all().all():
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\n--- Chi-Square Test Results ---")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.5f}")
    
    if p < 0.05:
        print("Result: Statistically significant association found between Target Group and Harm Status.")
        # Interpret direction
        prot_realized = props.loc['Protected Class', 'Realized Harm']
        gen_realized = props.loc['General Public/Other', 'Realized Harm']
        if prot_realized > gen_realized:
            print(f"Interpretation: Protected Classes face a HIGHER rate of realized harm ({prot_realized:.1f}%) compared to General Public ({gen_realized:.1f}%).")
        else:
            print(f"Interpretation: Protected Classes face a LOWER rate of realized harm ({prot_realized:.1f}%) compared to General Public ({gen_realized:.1f}%).")
    else:
        print("Result: No statistically significant association found.")
else:
    print("\nInsufficient data for reliable Chi-Square test (cell counts < 5 or empty).")

# 6. Visualization
plt.figure(figsize=(10, 6))
# Plot stacked bar chart of proportions
ax = props.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], rot=0)
plt.title('Realized vs Potential AI Harm by Target Group')
plt.ylabel('Percentage of Incidents')
plt.xlabel('Target Group')
plt.legend(title='Harm Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add percentage labels
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center')

plt.tight_layout()
plt.show()

print("\nExperiment Completed Successfully.")