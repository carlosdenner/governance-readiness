import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind, shapiro
import numpy as np
import os

# [debug] Check current directory files to ensure correct path
# print("Current directory files:", os.listdir('.'))

def load_dataset(filename):
    # Try current directory first
    if os.path.exists(filename):
        return pd.read_csv(filename)
    # Try one level up
    elif os.path.exists(os.path.join('..', filename)):
        return pd.read_csv(os.path.join('..', filename))
    else:
        raise FileNotFoundError(f"{filename} not found in . or ..")

try:
    # 1. Load the dataset
    df = load_dataset('step3_incident_coding.csv')
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # 2. Define Groups
    # Security group: 'security', 'supply_chain'
    # Other group: everything else
    security_harms = ['security', 'supply_chain']
    
    # Handle missing harm_type if any
    df['harm_type'] = df['harm_type'].fillna('unknown')
    
    df['harm_category'] = df['harm_type'].apply(lambda x: 'Security' if x in security_harms else 'Other')
    
    # 3. Extract Technique Counts
    # If technique_count is missing or 0, verify with techniques_used column count
    # The column 'technique_count' exists in metadata, but let's be robust
    if 'technique_count' not in df.columns:
        df['technique_count'] = df['techniques_used'].astype(str).apply(lambda x: len(x.split(';')) if x.lower() != 'nan' else 0)
    
    security_counts = df[df['harm_category'] == 'Security']['technique_count'].dropna()
    other_counts = df[df['harm_category'] == 'Other']['technique_count'].dropna()
    
    # 4. Descriptive Statistics
    print("\n=== Descriptive Statistics ===")
    print(f"Security Group (n={len(security_counts)}):")
    print(f"  Mean: {security_counts.mean():.2f}")
    print(f"  Median: {security_counts.median():.2f}")
    print(f"  Std Dev: {security_counts.std():.2f}")
    
    print(f"\nOther Group (n={len(other_counts)}):")
    print(f"  Mean: {other_counts.mean():.2f}")
    print(f"  Median: {other_counts.median():.2f}")
    print(f"  Std Dev: {other_counts.std():.2f}")
    
    # 5. Statistical Testing
    # Check normality
    # Shapiro-Wilk test requires N >= 3 usually. 
    if len(security_counts) >= 3 and len(other_counts) >= 3:
        _, p_norm_sec = shapiro(security_counts)
        _, p_norm_oth = shapiro(other_counts)
        print("\n=== Normality Tests (Shapiro-Wilk) ===")
        print(f"Security: p={p_norm_sec:.4f}")
        print(f"Other:    p={p_norm_oth:.4f}")
    
    # Use Mann-Whitney U test (non-parametric) as counts are often non-normal or samples small
    # T-test is also calculated for reference
    u_stat, p_mann = mannwhitneyu(security_counts, other_counts, alternative='two-sided')
    t_stat, p_ttest = ttest_ind(security_counts, other_counts, equal_var=False)
    
    print("\n=== Hypothesis Tests ===")
    print(f"Mann-Whitney U Test: U={u_stat}, p={p_mann:.4f}")
    print(f"Welch's T-test:      t={t_stat:.4f}, p={p_ttest:.4f}")
    
    if p_mann < 0.05:
        print("\nResult: Statistically significant difference in technique counts (p < 0.05).")
    else:
        print("\nResult: No statistically significant difference found (p >= 0.05).")

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    data_to_plot = [security_counts, other_counts]
    
    # Create boxplot
    box = plt.boxplot(data_to_plot, patch_artist=True, labels=['Security', 'Other'], zorder=3)
    
    # Customize colors
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add swarm/jitter plot
    x_vals_sec = np.random.normal(1, 0.04, size=len(security_counts))
    x_vals_oth = np.random.normal(2, 0.04, size=len(other_counts))
    plt.scatter(x_vals_sec, security_counts, alpha=0.6, color='blue', s=20, zorder=4)
    plt.scatter(x_vals_oth, other_counts, alpha=0.6, color='green', s=20, zorder=4)

    plt.title('Attack Complexity: Technique Counts by Harm Category')
    plt.ylabel('Number of Techniques Used')
    plt.xlabel('Harm Category')
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Add annotation of n counts
    plt.text(1, security_counts.max(), f"n={len(security_counts)}", ha='center', va='bottom')
    plt.text(2, other_counts.max(), f"n={len(other_counts)}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
