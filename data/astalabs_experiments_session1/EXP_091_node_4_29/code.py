import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

def load_data(filename):
    # Try current directory first
    if os.path.exists(filename):
        return pd.read_csv(filename)
    # Try parent directory
    elif os.path.exists(f"../{filename}"):
        return pd.read_csv(f"../{filename}")
    else:
        raise FileNotFoundError(f"{filename} not found in current or parent directory.")

try:
    # 1. Load Data
    df = load_data('step2_crosswalk_matrix.csv')
    print(f"Dataset loaded. Shape: {df.shape}")

    # 2. Group and Create Contingency Table
    # We want to see how 'source' relates to 'bundle'
    contingency = pd.crosstab(df['source'], df['bundle'])
    print("\nContingency Table (Source vs Bundle):")
    print(contingency)

    # 3. Statistical Test (Chi-Square)
    chi2, p, dof, expected = chi2_contingency(contingency)
    print("\n--- Chi-Square Test of Independence ---")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.6f}")
    print(f"Degrees of Freedom: {dof}")
    
    alpha = 0.05
    if p < alpha:
        print("\nResult: Reject Null Hypothesis (Significant Association)")
        print("The source framework significantly influences the distribution of Trust vs Integration requirements.")
    else:
        print("\nResult: Fail to Reject Null Hypothesis (No Significant Association)")

    # 4. Visualizations
    # Heatmap of counts
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency, annot=True, cmap="YlGnBu", fmt='d')
    plt.title("Heatmap: Governance Source vs. Competency Bundle")
    plt.ylabel("Source Framework")
    plt.xlabel("Readiness Bundle")
    plt.tight_layout()
    plt.show()

    # Normalized Stacked Bar Chart (to see proportions)
    # Normalize by row (Source) to compare proportions
    contingency_norm = contingency.div(contingency.sum(axis=1), axis=0)
    
    ax = contingency_norm.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', edgecolor='black')
    plt.title("Proportion of Readiness Bundles by Source Framework")
    plt.ylabel("Proportion")
    plt.xlabel("Source Framework")
    plt.legend(title='Bundle', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")