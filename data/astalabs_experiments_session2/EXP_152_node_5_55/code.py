import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load the dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback for local testing or different directory structure if needed, though instruction is explicit
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# --- 1. Map Autonomy Level ---
# Based on AIID taxonomy: Autonomy1 = Human-in-the-loop, Autonomy2 = Human-on-the-loop, Autonomy3 = Human-out-of-the-loop (High)
# Hypothesis groups "Low/Human-in-the-loop" vs "High"
def map_autonomy_code(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    
    if val_str == 'Autonomy3':
        return 'High Autonomy'
    elif val_str in ['Autonomy1', 'Autonomy2']:
        return 'Low Autonomy'
    else:
        return np.nan

aiid_df['Autonomy_Class'] = aiid_df['Autonomy Level'].apply(map_autonomy_code)

# --- 2. Categorize Technical Failures ---
def map_failure_type_refined(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    
    # Extended Robustness Keywords
    robustness_keys = [
        'robustness', 'adversarial', 'distribution', 'drift', 'generalization', 
        'model error', 'algorithm', 'prediction', 'classification', 
        'precision', 'recall', 'accuracy', 'reliability', 'bias',
        'context', 'misidentification', 'hallucination', 'generation hazard', 
        'unsafe exposure' # Often relates to model outputting unsafe content (robustness/alignment)
    ]
    
    # Operator/Human Error Keywords
    operator_keys = [
        'operator', 'human', 'user', 'configuration', 'setup', 'mistake', 
        'accidental', 'process', 'procedure'
    ]
    
    is_robustness = any(k in val_str for k in robustness_keys)
    is_operator = any(k in val_str for k in operator_keys)
    
    if is_robustness and not is_operator:
        return 'Robustness Failure'
    elif is_operator and not is_robustness:
        return 'Operator Error'
    elif is_robustness and is_operator:
        return 'Mixed/Ambiguous'
    else:
        return 'Other'

aiid_df['Failure_Category'] = aiid_df['Known AI Technical Failure'].apply(map_failure_type_refined)

# --- 3. Prepare Analysis Dataframe ---
# Filter for defined Autonomy and relevant Failure Categories
analysis_df = aiid_df.dropna(subset=['Autonomy_Class', 'Failure_Category'])

print("\n--- Failure Category Distribution (Before Filtering for Hypothesis) ---")
print(analysis_df['Failure_Category'].value_counts())

# Filter for specific hypothesis categories
final_df = analysis_df[analysis_df['Failure_Category'].isin(['Robustness Failure', 'Operator Error'])]

print("\n--- Final Analysis Dataset Summary ---")
print(final_df.groupby(['Autonomy_Class', 'Failure_Category']).size())

# --- 4. Statistical Test (Chi-Square) ---
contingency_table = pd.crosstab(final_df['Autonomy_Class'], final_df['Failure_Category'])

if contingency_table.empty or contingency_table.shape != (2, 2):
    print("\nInsufficient data for 2x2 Chi-Square test after filtering.")
    print("Contingency Table:\n", contingency_table)
else:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Visualization
    contingency_pct = contingency_table.div(contingency_table.sum(1), axis=0) * 100
    ax = contingency_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title('Distribution of Failure Types by Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Percentage of Incidents (%)')
    plt.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
    plt.tight_layout()
    plt.show()
