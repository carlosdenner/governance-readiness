import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define file name
file_name = 'step2_crosswalk_matrix.csv'

# Try to locate the file (current dir or parent dir)
if os.path.exists(file_name):
    file_path = file_name
elif os.path.exists(os.path.join('..', file_name)):
    file_path = os.path.join('..', file_name)
else:
    # Fallback to absolute path check or list dir for debugging if needed, 
    # but for now assume it's in the current dir based on previous success.
    file_path = file_name

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
    
    # Target columns
    col_hitl = 'Human-in-the-Loop Approval Gates'
    col_nondeter = 'Nondeterminism Controls & Output Validation'
    
    # Clean and create binary flags (assuming 'X' indicates presence)
    # We treat NaN as 0 (False) and 'X' (or any non-empty string) as 1 (True)
    df['Requires_HITL'] = df[col_hitl].fillna('').apply(lambda x: 1 if str(x).strip().upper() == 'X' else 0)
    df['Requires_Nondeterminism'] = df[col_nondeter].fillna('').apply(lambda x: 1 if str(x).strip().upper() == 'X' else 0)
    
    # --- Analysis 1: Human-in-the-Loop Approval Gates ---
    print(f"\n=== Analysis: {col_hitl} ===")
    contingency_hitl = pd.crosstab(df['bundle'], df['Requires_HITL'])
    print("Contingency Table (Bundle vs HITL):")
    print(contingency_hitl)
    
    chi2_hitl, p_hitl, dof_hitl, ex_hitl = stats.chi2_contingency(contingency_hitl)
    print(f"Chi-square statistic: {chi2_hitl:.4f}, p-value: {p_hitl:.4f}")
    
    # --- Analysis 2: Nondeterminism Controls ---
    print(f"\n=== Analysis: {col_nondeter} ===")
    contingency_nd = pd.crosstab(df['bundle'], df['Requires_Nondeterminism'])
    print("Contingency Table (Bundle vs Nondeterminism):")
    print(contingency_nd)
    
    chi2_nd, p_nd, dof_nd, ex_nd = stats.chi2_contingency(contingency_nd)
    print(f"Chi-square statistic: {chi2_nd:.4f}, p-value: {p_nd:.4f}")
    
    # --- Visualization ---
    # Calculate percentage of requirements in each bundle that have the control
    # Group by bundle, calculate mean of binary flag, multiply by 100
    summary = df.groupby('bundle')[['Requires_HITL', 'Requires_Nondeterminism']].mean() * 100
    
    print("\nPercentage of Requirements triggering control per Bundle:")
    print(summary)
    
    ax = summary.plot(kind='bar', figsize=(10, 6), rot=0, color=['skyblue', 'lightgreen'])
    plt.title('Association of Controls with Competency Bundles')
    plt.ylabel('Percentage of Requirements (%)')
    plt.xlabel('Competency Bundle')
    plt.legend(['Human-in-the-Loop (HITL)', 'Nondeterminism Controls'])
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (x + width/2, y + height + 1), 
                        ha='center')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")