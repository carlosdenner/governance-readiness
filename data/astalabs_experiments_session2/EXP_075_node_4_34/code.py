import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def run_experiment():
    # Load dataset
    file_path = "../astalabs_discovery_all_data.csv"
    
    print("Loading dataset...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print("File not found in parent directory, trying current directory.")
        df = pd.read_csv("astalabs_discovery_all_data.csv", low_memory=False)

    # Filter EO 13960
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 records: {len(df_eo)}")

    # Columns
    date_col = '18_date_initiated'
    catalog_col = '31_data_catalog'

    if date_col not in df_eo.columns or catalog_col not in df_eo.columns:
        print(f"Error: Columns {date_col} or {catalog_col} not found.")
        return

    # Check unique values in catalog
    print(f"Unique values in '{catalog_col}': {df_eo[catalog_col].unique()}")

    # Clean catalog: 1 if YES, 0 otherwise
    # Being conservative: treat NaN as 0 (No). Only explicit 'Yes' counts as having a catalog.
    def is_affirmative(val):
        if pd.isna(val): return 0
        s = str(val).lower().strip()
        return 1 if s == 'yes' or s == 'true' else 0

    df_eo['has_catalog'] = df_eo[catalog_col].apply(is_affirmative)

    # Clean date
    # Convert to datetime using coerce to handle mixed formats
    df_eo['dt'] = pd.to_datetime(df_eo[date_col], errors='coerce')
    
    # Check how many dates were parsed
    parsed_count = df_eo['dt'].notna().sum()
    print(f"Parsed {parsed_count} valid dates out of {len(df_eo)} records.")

    if parsed_count < 10:
        print("Too few valid dates parsed. Aborting analysis.")
        print("Sample raw dates:", df_eo[date_col].dropna().head(10).tolist())
        return

    df_clean = df_eo.dropna(subset=['dt']).copy()
    
    # Define Cohorts
    # Legacy: Started before 2019-01-01
    cutoff = pd.Timestamp("2019-01-01")
    df_clean['cohort'] = df_clean['dt'].apply(lambda x: 'Legacy' if x < cutoff else 'Modern')

    # Stats
    stats_df = df_clean.groupby('cohort')['has_catalog'].agg(['count', 'mean', 'sum'])
    stats_df.rename(columns={'mean': 'compliance_rate', 'sum': 'num_compliant', 'count': 'n'}, inplace=True)
    
    print("\n--- Cohort Analysis ---")
    print(stats_df)

    # Contingency Table for Chi-Square
    # We need a 2x2 table of counts
    contingency = pd.crosstab(df_clean['cohort'], df_clean['has_catalog'])
    print("\n--- Contingency Table (0=No, 1=Yes) ---")
    print(contingency)

    # Run Chi-Square Test of Independence
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test: statistic={chi2:.4f}, p-value={p_val:.4f}")

    # Calculate Odds Ratio manually for interpretation
    # OR = (Modern_Yes / Modern_No) / (Legacy_Yes / Legacy_No)
    if 'Legacy' in contingency.index and 'Modern' in contingency.index:
        try:
            # Check if columns 0 and 1 exist (0=No, 1=Yes)
            leg_no = contingency.loc['Legacy', 0] if 0 in contingency.columns else 0
            leg_yes = contingency.loc['Legacy', 1] if 1 in contingency.columns else 0
            mod_no = contingency.loc['Modern', 0] if 0 in contingency.columns else 0
            mod_yes = contingency.loc['Modern', 1] if 1 in contingency.columns else 0
            
            if mod_no > 0 and leg_yes > 0 and leg_no > 0:
                 or_val = (mod_yes / mod_no) / (leg_yes / leg_no)
                 print(f"Odds Ratio (Modern vs Legacy): {or_val:.4f}")
            else:
                 print("Cannot calculate Odds Ratio due to zero counts in denominator.")
        except Exception as e:
            print(f"Could not calculate Odds Ratio: {e}")

    # Visualization
    df_clean['year'] = df_clean['dt'].dt.year
    # Group by year
    yearly = df_clean.groupby('year')['has_catalog'].mean()
    counts = df_clean.groupby('year')['has_catalog'].count()
    
    # Filter years with fewer than 5 records to reduce noise in the plot
    valid_years = counts[counts >= 5].index
    if len(valid_years) > 0:
        yearly_plot = yearly.loc[valid_years].sort_index()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(yearly_plot.index, yearly_plot.values, color='cornflowerblue', label='Data Catalog Rate')
        
        # Add values on bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)

        plt.axvline(x=2018.5, color='red', linestyle='--', linewidth=2, label='2019 Cutoff')
        plt.title('Data Catalog Compliance Rate by Project Start Year')
        plt.xlabel('Year Initiated')
        plt.ylabel('Proportion with Data Catalog')
        plt.ylim(0, 1.15)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.show()
    else:
        print("Not enough data per year to plot.")

if __name__ == "__main__":
    run_experiment()