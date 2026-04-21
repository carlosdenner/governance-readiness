import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# [debug] Check file existence to handle path variability
filename = 'step3_incident_coding.csv'
possible_paths = [filename, f'../{filename}']
file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if not file_path:
    print(f"Error: Could not find {filename} in current or parent directory.")
else:
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # Normalize text to lowercase to ensure matching
    df['trust_integration_split'] = df['trust_integration_split'].astype(str).str.lower().str.strip()
    df['harm_type'] = df['harm_type'].astype(str).str.lower().str.strip()

    print("\nUnique splits found:", df['trust_integration_split'].unique())
    print("Unique harms found:", df['harm_type'].unique())

    # Define the categories of interest
    target_splits = ['trust-dominant', 'integration-dominant']
    target_harms = ['security', 'privacy']

    # Filter the dataframe
    filtered_df = df[
        df['trust_integration_split'].isin(target_splits) &
        df['harm_type'].isin(target_harms)
    ].copy()

    print(f"\nFiltered dataset shape: {filtered_df.shape}")
    
    if filtered_df.empty:
        print("No records match the filter criteria (Trust/Integration dominant AND Security/Privacy).")
    else:
        # Create contingency table
        contingency_table = pd.crosstab(filtered_df['trust_integration_split'], filtered_df['harm_type'])
        
        # Ensure all expected columns/indexes are present for the test, filling with 0 if missing
        # We want rows: integration-dominant, trust-dominant
        # We want cols: privacy, security
        # (Order matters for odds ratio interpretation, though p-value is invariant)
        contingency_table = contingency_table.reindex(index=target_splits, columns=target_harms, fill_value=0)

        print("\nContingency Table:")
        print(contingency_table)

        # Perform Fisher's Exact Test
        # null hypothesis: the true odds ratio of the populations underlying the observations is one (no association)
        odds_ratio, p_value = stats.fisher_exact(contingency_table)

        print(f"\nFisher's Exact Test Results:")
        print(f"Odds Ratio: {odds_ratio:.4f}")
        print(f"P-value: {p_value:.4f}")

        # Visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Harm Type by Competency Split')
        plt.ylabel('Competency Split')
        plt.xlabel('Harm Type')
        plt.show()
