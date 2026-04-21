import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# [debug] Check for file existence in parent directory
filename = 'step3_incident_coding.csv'
parent_path = os.path.join('..', filename)
local_path = filename

if os.path.exists(parent_path):
    filepath = parent_path
elif os.path.exists(local_path):
    filepath = local_path
else:
    # Fallback to creating a dummy dataset if file is missing (for robust execution in unknown envs)
    # However, per instructions, I should assume dataset is available.
    print(f"File {filename} not found in . or ..")
    filepath = None

if filepath:
    df = pd.read_csv(filepath)
    print(f"Successfully loaded {filepath}")
    print(f"Dataset shape: {df.shape}")

    # 2. Group by 'failure_mode' and analyze 'technique_count'
    if 'failure_mode' in df.columns and 'technique_count' in df.columns:
        
        # Descriptive Statistics
        print("\n=== Technique Count Statistics by Failure Mode ===")
        stats_summary = df.groupby('failure_mode')['technique_count'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        print(stats_summary)

        # Prepare data for Statistical Test
        groups = []
        group_names = []
        for name, group in df.groupby('failure_mode'):
            groups.append(group['technique_count'].values)
            group_names.append(name)

        # 3. Perform Kruskal-Wallis Test
        # We use Kruskal-Wallis because sample sizes are likely very unequal and distributions may not be normal.
        if len(groups) > 1:
            h_stat, p_val = stats.kruskal(*groups)
            print("\n=== Kruskal-Wallis H-test ===")
            print(f"H-statistic: {h_stat:.4f}")
            print(f"P-value: {p_val:.4f}")
            
            alpha = 0.05
            if p_val < alpha:
                print("Conclusion: Significant difference in attack complexity between failure modes.")
            else:
                print("Conclusion: No significant difference in attack complexity between failure modes.")
        else:
            print("\nNot enough failure mode groups to perform statistical testing.")

        # 4. Visualization
        plt.figure(figsize=(10, 6))
        plt.boxplot(groups, labels=group_names)
        plt.title('Distribution of Attack Complexity (Technique Count) by Failure Mode')
        plt.ylabel('Technique Count (Complexity)')
        plt.xlabel('Failure Mode')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
    else:
        print("Required columns 'failure_mode' or 'technique_count' not found.")
