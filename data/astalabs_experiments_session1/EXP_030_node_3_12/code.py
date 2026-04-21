import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Define the file path (using current directory based on previous error)
file_path = 'step2_crosswalk_matrix.csv'

try:
    # 1. Load the dataset
    df = pd.read_csv(file_path)

    # 2. Identify Metadata and Control Columns
    # Metadata columns based on dataset description
    metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
    
    # All other columns are architecture controls
    control_cols = [c for c in df.columns if c not in metadata_cols]
    
    # 3. Calculate 'Control Density'
    # Count 'X' (case-insensitive) in control columns for each row
    def calculate_density(row):
        count = 0
        for col in control_cols:
            val = str(row[col]).strip().upper()
            if val == 'X':
                count += 1
        return count

    df['control_density'] = df.apply(calculate_density, axis=1)

    # 4. Create Grouping Variables
    # Group 1: NIST (NIST AI RMF 1.0, NIST GenAI Profile)
    nist_df = df[df['source'].str.contains('NIST', case=False, na=False)]
    
    # Group 2: EU AI Act
    eu_df = df[df['source'].str.contains('EU AI Act', case=False, na=False)]

    # Extract density series
    nist_density = nist_df['control_density']
    eu_density = eu_df['control_density']

    # 5. Statistical Analysis
    nist_mean = nist_density.mean()
    nist_std = nist_density.std()
    nist_n = len(nist_density)

    eu_mean = eu_density.mean()
    eu_std = eu_density.std()
    eu_n = len(eu_density)

    # Independent samples t-test (Welch's t-test for unequal variances)
    t_stat, p_val = stats.ttest_ind(nist_density, eu_density, equal_var=False)

    # Print Statistical Results
    print("=== Control Density Analysis: NIST vs EU AI Act ===")
    print(f"NIST Group (n={nist_n}): Mean = {nist_mean:.4f}, Std = {nist_std:.4f}")
    print(f"EU Group   (n={eu_n}): Mean = {eu_mean:.4f}, Std = {eu_std:.4f}")
    print(f"Difference in Means: {nist_mean - eu_mean:.4f}")
    print(f"T-Test Results: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
    
    alpha = 0.05
    if p_val < alpha:
        print("Result: Statistically Significant Difference (Reject Null Hypothesis)")
    else:
        print("Result: No Statistically Significant Difference (Fail to Reject Null Hypothesis)")

    # 6. Visualization
    # Prepare data for plotting
    groups = ['NIST Family', 'EU AI Act']
    means = [nist_mean, eu_mean]
    # Calculate Standard Error of the Mean (SEM) for error bars
    sems = [stats.sem(nist_density), stats.sem(eu_density)]

    plt.figure(figsize=(8, 6))
    # Create bar chart with error bars
    bars = plt.bar(groups, means, yerr=sems, capsize=10, color=['#4C72B0', '#55A868'], alpha=0.9, width=0.6)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.1, 
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Mean Control Density (Mapped Controls per Req)')
    plt.title('Comparison of Technical Prescription: NIST vs EU AI Act')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add significance annotation line if significant
    if p_val < 0.05:
        x1, x2 = 0, 1
        y_max = max(means) + max(sems) + 0.5
        h = 0.1
        plt.plot([x1, x1, x2, x2], [y_max, y_max+h, y_max+h, y_max], lw=1.5, c='k')
        plt.text((x1+x2)*.5, y_max+h, f"p={p_val:.3f}", ha='center', va='bottom', color='k')

    plt.ylim(0, max(means) + max(sems) + 1.5)  # Adjust y-axis limit for labels
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
