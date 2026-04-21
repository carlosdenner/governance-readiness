import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_experiment():
    # Attempt to locate the file in current directory or parent directory
    filename = 'astalabs_discovery_all_data.csv'
    if os.path.exists(filename):
        file_path = filename
    elif os.path.exists(f'../{filename}'):
        file_path = f'../{filename}'
    else:
        print(f"Error: Could not find {filename} in current or parent directory.")
        return

    print(f"Loading dataset from {file_path}...")
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter for EO13960 subset
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 subset size: {len(df_eo)} rows")

    # Columns of interest
    col_pii = '29_contains_pii'
    col_catalog = '31_data_catalog'

    if col_pii not in df_eo.columns or col_catalog not in df_eo.columns:
        print(f"Required columns not found. Available: {df_eo.columns.tolist()}")
        return

    # Data Cleaning
    def clean_response(val):
        s = str(val).strip().upper()
        if s in ['YES', 'Y', 'TRUE', '1']:
            return 'Yes'
        elif s in ['NO', 'N', 'FALSE', '0']:
            return 'No'
        return None

    df_eo['has_pii'] = df_eo[col_pii].apply(clean_response)
    df_eo['has_catalog'] = df_eo[col_catalog].apply(clean_response)

    # Drop invalid rows for analysis
    df_clean = df_eo.dropna(subset=['has_pii', 'has_catalog'])
    print(f"Valid records for analysis (after cleaning): {len(df_clean)}")

    if len(df_clean) == 0:
        print("Insufficient data for analysis.")
        return

    # 1. Contingency Table
    contingency = pd.crosstab(df_clean['has_pii'], df_clean['has_catalog'])
    print("\n--- Contingency Table (Counts) ---")
    print(contingency)

    # 2. Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # 3. Odds Ratio Calculation
    try:
        def get_val(r, c):
            return contingency.loc[r, c] if r in contingency.index and c in contingency.columns else 0

        a = get_val('Yes', 'Yes') # PII=Yes, Cat=Yes
        b = get_val('Yes', 'No')  # PII=Yes, Cat=No
        c = get_val('No', 'Yes')  # PII=No, Cat=Yes
        d = get_val('No', 'No')   # PII=No, Cat=No
        
        print(f"\nCounts used for OR: PII_Yes/Cat_Yes={a}, PII_Yes/Cat_No={b}, PII_No/Cat_Yes={c}, PII_No/Cat_No={d}")

        if b == 0 or c == 0:
            print("Zero count detected in denominator, using Haldane-Anscombe correction (+0.5).")
            odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        else:
            odds_ratio = (a * d) / (b * c)
            
        print(f"Odds Ratio: {odds_ratio:.4f}")
        
    except Exception as e:
        print(f"Error calculating Odds Ratio: {e}")

    # Interpret results
    alpha = 0.05
    if p < alpha:
        print("\nResult: Statistically SIGNIFICANT association found.")
    else:
        print("\nResult: NO statistically significant association found.")

    # 4. Visualization
    # Prepare data for plotting (percentages)
    # Group by PII status and Catalog status to get counts
    plot_data = df_clean.groupby(['has_pii', 'has_catalog']).size().reset_index(name='count')
    
    # Calculate totals per PII group to normalize percentages
    # transform('sum') broadcasts the sum back to the original rows of the group
    plot_data['total_in_group'] = plot_data.groupby('has_pii')['count'].transform('sum')
    
    # Calculate percentage
    plot_data['percent'] = (plot_data['count'] / plot_data['total_in_group']) * 100

    print("\nPlot Data Preview:")
    print(plot_data)

    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(data=plot_data, x='has_pii', y='percent', hue='has_catalog', palette='viridis')
    
    plt.title('Data Catalog Implementation by PII Status (EO13960)')
    plt.xlabel('System Processes PII?')
    plt.ylabel('Percentage of Systems within Group (%)')
    plt.legend(title='Has Data Catalog')
    plt.ylim(0, 100)
    
    # Add labels
    for container in barplot.containers:
        barplot.bar_label(container, fmt='%.1f%%', padding=3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()