import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_experiment():
    filename = 'step3_incident_coding.csv'
    # Try current directory first, then parent if not found (robustness)
    if not os.path.exists(filename):
        if os.path.exists(f'../{filename}'):
            filename = f'../{filename}'
        else:
            print(f"Error: {filename} not found in . or ..")
            return

    try:
        # 1. Load the dataset
        df = pd.read_csv(filename)
        print(f"Dataset loaded from {filename}. Rows: {len(df)}")
        
        # 2. Preprocess / Map categories
        # Map failure_mode
        def map_failure(mode):
            if pd.isna(mode):
                return 'Other'
            mode = str(mode).lower().strip()
            if 'prevention' in mode:
                return 'Prevention'
            elif 'detection' in mode or 'response' in mode:
                return 'Detection/Response'
            else:
                return 'Other'

        df['failure_category'] = df['failure_mode'].apply(map_failure)
        
        # Normalize split
        df['trust_integration_split'] = df['trust_integration_split'].astype(str).str.lower().str.strip()
        
        # 3. Filter for specific splits
        target_splits = ['trust-dominant', 'integration-dominant']
        df_filtered = df[df['trust_integration_split'].isin(target_splits)].copy()
        
        print(f"Filtered dataset size: {len(df_filtered)}")
        print("Split distribution in filtered set:")
        print(df_filtered['trust_integration_split'].value_counts())
        print("Failure category distribution in filtered set:")
        print(df_filtered['failure_category'].value_counts())

        if len(df_filtered) == 0:
            print("No data matching filter criteria. Cannot perform test.")
            return

        # 4. Create Contingency Table
        contingency_table = pd.crosstab(
            df_filtered['trust_integration_split'], 
            df_filtered['failure_category']
        )
        
        print("\nContingency Table:")
        print(contingency_table)

        # 5. Statistical Test
        # Check shape. If we have both rows and both columns, do Fisher.
        # If we are missing columns (e.g. only Prevention), we can't do Fisher test of independence easily 2x2.
        
        row_count, col_count = contingency_table.shape
        
        if row_count == 2 and col_count == 2:
            odds_ratio, p_value = stats.fisher_exact(contingency_table)
            print(f"\nFisher's Exact Test Results:")
            print(f"Odds Ratio: {odds_ratio}")
            print(f"P-value: {p_value}")
        else:
            print("\nContingency table is not 2x2 (likely due to zero counts in one category). Cannot perform standard 2x2 Fisher's Exact Test.")
            print(f"Shape is {contingency_table.shape}")

        # 6. Visualization
        if not contingency_table.empty:
            # Align columns for consistent coloring
            # We want 'Prevention' and 'Detection/Response' if they exist
            cols_to_plot = []
            if 'Prevention' in contingency_table.columns:
                cols_to_plot.append('Prevention')
            if 'Detection/Response' in contingency_table.columns:
                cols_to_plot.append('Detection/Response')
            
            if cols_to_plot:
                ax = contingency_table[cols_to_plot].plot(kind='bar', stacked=True, figsize=(8, 6), color=['skyblue', 'salmon'])
                plt.title('Failure Mode Distribution by Competency Split')
                plt.xlabel('Competency Split')
                plt.ylabel('Count')
                plt.xticks(rotation=0)
                plt.legend(title='Failure Category')
                plt.tight_layout()
                plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_experiment()