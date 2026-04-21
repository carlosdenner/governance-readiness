import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import os

# Ensure statsmodels is installed
try:
    from statsmodels.stats.proportion import proportions_ztest
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "statsmodels"])
    from statsmodels.stats.proportion import proportions_ztest

def run_experiment():
    try:
        # Attempt to find the dataset
        filename = 'astalabs_discovery_all_data.csv'
        if os.path.exists(filename):
            filepath = filename
        elif os.path.exists(f'../{filename}'):
            filepath = f'../{filename}'
        else:
            # If not found, list current directory to help debug, though we must fail eventually
            print("Dataset not found in current or parent directory.")
            return

        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        
        # Filter for the relevant source table
        df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
        print(f"Total EO 13960 records: {len(df_eo)}")

        # 1. Parse '20_date_implemented' to extract year
        # Standardize date format handling
        df_eo['impl_date'] = pd.to_datetime(df_eo['20_date_implemented'], errors='coerce')
        
        # Drop rows where date is unknown
        df_clean = df_eo.dropna(subset=['impl_date']).copy()
        df_clean['impl_year'] = df_clean['impl_date'].dt.year
        
        # Filter for valid years (e.g., 1990-2025) to remove data entry errors
        df_clean = df_clean[(df_clean['impl_year'] >= 1990) & (df_clean['impl_year'] <= 2025)]
        print(f"Records with valid implementation dates (1990-2025): {len(df_clean)}")

        # 2. Parse '62_disparity_mitigation' into binary
        # Metadata: '62_disparity_mitigation' contains text describing mitigation or 'No', 'N/A', etc.
        def parse_mitigation(val):
            if pd.isna(val):
                return 0
            val_str = str(val).lower().strip()
            if not val_str:
                return 0
            # List of values indicating absence of control
            negatives = ['no', 'n/a', 'none', 'not applicable', '0', 'false', 'unknown', 'tbd', 'not evaluated', 'NaN']
            if val_str in negatives:
                return 0
            # If it contains text not in negatives, assume it describes a mitigation
            return 1

        df_clean['has_mitigation'] = df_clean['62_disparity_mitigation'].apply(parse_mitigation)

        # 3. Split into Pre-2021 and Post-2021 (EO 13960 was late 2020, usually taking effect 2021 for this analysis)
        # The hypothesis specifies "Post-2021", usually meaning >= 2021.
        pre_2021 = df_clean[df_clean['impl_year'] < 2021]
        post_2021 = df_clean[df_clean['impl_year'] >= 2021]

        n_pre = len(pre_2021)
        count_pre = pre_2021['has_mitigation'].sum()
        prop_pre = count_pre / n_pre if n_pre > 0 else 0

        n_post = len(post_2021)
        count_post = post_2021['has_mitigation'].sum()
        prop_post = count_post / n_post if n_post > 0 else 0

        print("\n--- Comparative Analysis (Cutoff: 2021) ---")
        print(f"Pre-2021 Systems (Legacy): n={n_pre}")
        print(f"  Bias Mitigation Compliance: {count_pre} ({prop_pre:.2%})")
        print(f"Post-2021 Systems (Modern): n={n_post}")
        print(f"  Bias Mitigation Compliance: {count_post} ({prop_post:.2%})")

        # 4. Perform Z-test
        if n_pre > 0 and n_post > 0:
            count = np.array([count_pre, count_post])
            nobs = np.array([n_pre, n_post])
            stat, pval = proportions_ztest(count, nobs, alternative='two-sided')
            print(f"\nZ-Test Results:")
            print(f"  Z-statistic: {stat:.4f}")
            print(f"  P-value: {pval:.4e}")
            
            if pval < 0.05:
                print("  Conclusion: Statistically Significant Difference.")
            else:
                print("  Conclusion: No Statistically Significant Difference.")
        else:
            print("\nInsufficient data for Z-test.")

        # 5. Visualization: Compliance Rate over Time
        # Group by year
        yearly_stats = df_clean.groupby('impl_year')['has_mitigation'].agg(['mean', 'count']).reset_index()
        
        # Filter to years with at least a few systems to avoid noisy spikes
        yearly_stats_plot = yearly_stats[yearly_stats['count'] >= 5]

        plt.figure(figsize=(10, 6))
        plt.plot(yearly_stats_plot['impl_year'], yearly_stats_plot['mean'], marker='o', linestyle='-', linewidth=2, label='Mitigation Rate')
        plt.axvline(x=2021, color='r', linestyle='--', label='EO 13960 Era (2021+)')
        
        plt.title('Temporal Adoption of Bias Mitigation Controls')
        plt.xlabel('Year Implemented')
        plt.ylabel('Proportion of Systems with Controls')
        plt.ylim(-0.05, 0.40) # Adjusted Y-limit based on likely low compliance rates to make chart readable
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_experiment()