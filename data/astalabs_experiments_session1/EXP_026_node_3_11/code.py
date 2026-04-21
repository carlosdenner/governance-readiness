import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

def load_dataset(filename):
    parent_path = os.path.join("..", filename)
    if os.path.exists(parent_path):
        return pd.read_csv(parent_path)
    elif os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        raise FileNotFoundError(f"Dataset {filename} not found.")

try:
    # 1. Load Datasets
    df_competencies = load_dataset('step1_sub_competencies.csv')
    df_coverage = load_dataset('step3_coverage_map.csv')

    # Clean column names
    df_competencies.columns = df_competencies.columns.str.strip()
    df_coverage.columns = df_coverage.columns.str.strip()

    # 2. Merge Datasets
    merged_df = pd.merge(
        df_competencies[['id', 'observable_practices', 'name']],
        df_coverage[['sub_competency_id', 'incident_count', 'coverage_status']],
        left_on='id',
        right_on='sub_competency_id',
        how='inner'
    )

    print(f"Merged dataset shape: {merged_df.shape}")

    # 3. Calculate 'practice_count' (Operational Complexity)
    # Assuming semicolon separation based on dataset metadata descriptions for similar fields
    def count_practices(text):
        if pd.isna(text):
            return 0
        # Split by semicolon, strip whitespace, and filter out empty strings
        items = [item for item in str(text).split(';') if item.strip()]
        return len(items)

    merged_df['practice_count'] = merged_df['observable_practices'].apply(count_practices)

    print("\nData Preview (ID, Practice Count, Incident Count):")
    print(merged_df[['id', 'practice_count', 'incident_count']])

    # 4. Calculate Spearman's Rank Correlation
    if len(merged_df) > 1 and merged_df['practice_count'].std() > 0 and merged_df['incident_count'].std() > 0:
        corr, p_value = spearmanr(merged_df['practice_count'], merged_df['incident_count'])
        print("\n=== Statistical Analysis ===")
        print(f"Spearman's Rank Correlation Coefficient: {corr:.4f}")
        print(f"P-value: {p_value:.4f}")
    else:
        corr, p_value = float('nan'), float('nan')
        print("\nCannot calculate correlation: insufficient data or zero variance.")

    # 5. Generate Scatter Plot
    plt.figure(figsize=(10, 6))
    
    # Using regplot to show the scatter and a linear regression fit to visualize the trend
    sns.regplot(
        x='practice_count',
        y='incident_count',
        data=merged_df,
        color='teal',
        ci=None, # Disable confidence interval shading for cleaner look with few points
        scatter_kws={'s': 100, 'alpha': 0.7}
    )

    # Annotate points with IDs
    for i, row in merged_df.iterrows():
        plt.text(
            row['practice_count'] + 0.1, 
            row['incident_count'] + 0.1, 
            row['id'], 
            fontsize=9
        )

    plt.title(f'Operational Complexity vs. Real-World Incident Frequency\n(Spearman r={corr:.2f}, p={p_value:.3f})')
    plt.xlabel('Operational Complexity (Count of Observable Practices)')
    plt.ylabel('Incident Frequency (Count of MITRE ATLAS Cases)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Force integer ticks for count data
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
