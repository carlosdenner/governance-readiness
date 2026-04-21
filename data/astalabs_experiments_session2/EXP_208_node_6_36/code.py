import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os

def analyze_adversarial_complexity():
    # Load dataset
    file_path = '../astalabs_discovery_all_data.csv'
    if not os.path.exists(file_path):
        file_path = 'astalabs_discovery_all_data.csv' # Fallback for local testing if needed

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}")
        return

    # Filter for ATLAS cases
    atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
    
    if atlas_df.empty:
        print("No ATLAS cases found in the dataset.")
        return

    print(f"Loaded {len(atlas_df)} ATLAS cases.")

    # Determine delimiter by inspecting a sample
    sample_tactics = atlas_df['tactics'].dropna().iloc[0] if not atlas_df['tactics'].dropna().empty else ""
    delimiter = '|' if '|' in str(sample_tactics) else ','
    print(f"Detected delimiter for tactics: '{delimiter}'")

    # Function to parse tactics and count length
    def parse_and_count(tactic_str):
        if pd.isna(tactic_str):
            return 0, False
        
        # Normalize and split
        t_str = str(tactic_str)
        items = [x.strip() for x in t_str.split(delimiter) if x.strip()]
        
        # Count unique tactics
        unique_items = list(set(items))
        chain_length = len(unique_items)
        
        # Check for 'Impact' tactic (case-insensitive)
        has_impact = any('impact' in item.lower() for item in unique_items)
        
        return chain_length, has_impact

    # Apply processing
    # Result is a DataFrame with two columns, which we assign back
    results = atlas_df['tactics'].apply(lambda x: parse_and_count(x))
    atlas_df['chain_length'] = results.apply(lambda x: x[0])
    atlas_df['achieved_impact'] = results.apply(lambda x: x[1])

    # Separate groups
    impact_group = atlas_df[atlas_df['achieved_impact'] == True]['chain_length']
    no_impact_group = atlas_df[atlas_df['achieved_impact'] == False]['chain_length']

    # Statistics
    n_impact = len(impact_group)
    n_no_impact = len(no_impact_group)
    
    print(f"Cases achieving Impact: {n_impact}")
    print(f"Cases NOT achieving Impact: {n_no_impact}")
    
    if n_impact == 0 or n_no_impact == 0:
        print("Cannot perform statistical test: One of the groups is empty.")
        return

    mean_impact = impact_group.mean()
    mean_no_impact = no_impact_group.mean()
    median_impact = impact_group.median()
    median_no_impact = no_impact_group.median()
    
    print(f"\nMean Chain Length (Impact): {mean_impact:.2f} (Median: {median_impact})")
    print(f"Mean Chain Length (No Impact): {mean_no_impact:.2f} (Median: {median_no_impact})")

    # Mann-Whitney U Test (Impact > No Impact)
    stat, p_value = mannwhitneyu(impact_group, no_impact_group, alternative='greater')
    print(f"\nMann-Whitney U Test results:")
    print(f"Statistic: {stat}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Conclusion: Statistically significant difference (p < 0.05). Hypothesis SUPPORTED.")
    else:
        print("Conclusion: No statistically significant difference (p >= 0.05). Hypothesis NOT SUPPORTED.")

    # Visualization
    plt.figure(figsize=(10, 6))
    # Create boxplot
    bp = plt.boxplot([no_impact_group, impact_group], 
                     labels=['No Impact', 'Impact Achieved'],
                     patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Adversarial Kill Chain Complexity: Impact vs. No Impact')
    plt.ylabel('Number of Unique Tactics (Chain Length)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add swarm plot or jitter for individual points if N is small
    y_no = no_impact_group
    x_no = np.random.normal(1, 0.04, size=len(y_no))
    plt.scatter(x_no, y_no, alpha=0.6, color='blue', s=20)

    y_yes = impact_group
    x_yes = np.random.normal(2, 0.04, size=len(y_yes))
    plt.scatter(x_yes, y_yes, alpha=0.6, color='green', s=20)

    plt.show()

if __name__ == "__main__":
    analyze_adversarial_complexity()