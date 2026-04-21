import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

def load_dataset(filename):
    # Try current directory first
    if os.path.exists(filename):
        return pd.read_csv(filename)
    # Try one level up
    parent_path = os.path.join('..', filename)
    if os.path.exists(parent_path):
        return pd.read_csv(parent_path)
    raise FileNotFoundError(f"Could not find {filename} in . or ..")

try:
    # 1. Load the dataset
    df = load_dataset('step2_competency_statements.csv')
    print(f"Dataset loaded successfully with shape: {df.shape}")

    # 2. Create variable 'citation_count'
    # Pattern to match citations like [#1], [#12]
    citation_pattern = r'\[#\d+\]'
    
    def count_citations(row):
        # Concatenate text from both columns to search for citations
        text = str(row['competency_statement']) + " " + str(row['evidence_summary'])
        # Find all occurrences
        matches = re.findall(citation_pattern, text)
        return len(matches)

    df['citation_count'] = df.apply(count_citations, axis=1)

    # 3. Create variable 'control_count'
    def count_controls(val):
        if pd.isna(val) or str(val).strip() == '':
            return 0
        # Split by semicolon, strip whitespace, filter out empty strings
        items = [x.strip() for x in str(val).split(';') if x.strip()]
        return len(items)

    df['control_count'] = df['applicable_controls'].apply(count_controls)

    # Print sample to verify
    print("\nSample of calculated counts:")
    print(df[['competency_id', 'citation_count', 'control_count']].head())

    # 4. Perform Correlation Tests
    # Pearson (Linear)
    pearson_r, pearson_p = stats.pearsonr(df['citation_count'], df['control_count'])
    # Spearman (Rank - robust to outliers/non-normal)
    spearman_rho, spearman_p = stats.spearmanr(df['citation_count'], df['control_count'])

    print("\n=== Correlation Results ===")
    print(f"Pearson Correlation (r): {pearson_r:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman Correlation (rho): {spearman_rho:.4f} (p-value: {spearman_p:.4f})")

    # Interpretation
    alpha = 0.05
    if pearson_p < alpha:
        print("Conclusion: Statistically significant linear correlation found.")
    else:
        print("Conclusion: No statistically significant linear correlation found.")

    # 5. Visualize
    plt.figure(figsize=(10, 6))
    sns.regplot(x='citation_count', y='control_count', data=df, 
                scatter_kws={'alpha':0.6, 's':60}, line_kws={'color':'red'})
    
    plt.title('Correlation: Literature Evidence vs. Technical Complexity')
    plt.xlabel('Citation Count (Evidence Volume)')
    plt.ylabel('Control Count (Architecture Fan-out)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add correlation text to plot
    plt.annotate(f'Pearson r={pearson_r:.2f} (p={pearson_p:.3f})',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
