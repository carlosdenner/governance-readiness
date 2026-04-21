import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def run_experiment():
    print("Starting 'Opaque Defense' Hypothesis Experiment...")
    
    # 1. Load the dataset
    try:
        # Use local path as previous attempts with '../' failed
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found at 'astalabs_discovery_all_data.csv'")
        return

    # 2. Filter for EO13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 Records: {len(df_eo)}")

    # 3. Categorize Agencies
    # Defense/Security keywords: DOD, DHS, DOJ, Defense, Homeland, Justice
    def categorize_agency(agency_name):
        if pd.isna(agency_name):
            return 'Civilian' # Default to civilian if unknown/missing
        name_lower = str(agency_name).lower()
        # Check for specific defense/security keywords
        security_keywords = ['defense', 'homeland security', 'justice', 'dod', 'dhs', 'doj']
        if any(k in name_lower for k in security_keywords):
            return 'Defense/Security'
        return 'Civilian'

    df_eo['agency_type'] = df_eo['3_agency'].apply(categorize_agency)
    
    print("\nAgency Type Distribution (All EO13960):")
    print(df_eo['agency_type'].value_counts())

    # 4. Filter for High Impact Systems
    # We look for 'Rights-Impacting', 'Safety-Impacting' in column '17_impact_type'
    # Just to be safe, we'll include 'high' if present, though usually it's Rights/Safety.
    
    def is_high_impact(val):
        if pd.isna(val):
            return False
        val_lower = str(val).lower()
        return 'rights' in val_lower or 'safety' in val_lower or 'high' in val_lower

    df_high = df_eo[df_eo['17_impact_type'].apply(is_high_impact)].copy()
    print(f"\nHigh Impact Systems Identified: {len(df_high)}")
    
    if len(df_high) == 0:
        print("No high impact systems found. Checking sample values of '17_impact_type':")
        print(df_eo['17_impact_type'].dropna().unique()[:5])
        return

    print("High Impact Agency Distribution:")
    print(df_high['agency_type'].value_counts())

    # 5. Calculate 'Opacity Rate'
    # Opacity defined as: 59_ai_notice == 'No' OR 67_opt_out == 'No'
    # We will normalize text to lowercase and strip whitespace.
    
    def check_opacity(row):
        # Get values, handle NaNs as empty strings
        notice = str(row.get('59_ai_notice', '')).strip().lower()
        opt_out = str(row.get('67_opt_out', '')).strip().lower()
        
        # If either specific transparency mechanism is explicitly denied ('no'), it is opaque.
        # Note: If data is missing (nan), we don't count it as 'No' unless we assume missing = opaque.
        # The prompt says "where ... is 'No'". So we strictly look for 'no'.
        is_opaque = (notice == 'no') or (opt_out == 'no')
        return 1 if is_opaque else 0

    df_high['is_opaque'] = df_high.apply(check_opacity, axis=1)

    # 6. Compare Opacity Rates
    stats_df = df_high.groupby('agency_type')['is_opaque'].agg(['count', 'sum', 'mean'])
    stats_df.columns = ['Total_Systems', 'Opaque_Systems', 'Opacity_Rate']
    
    print("\nOpacity Statistics by Agency Type (High Impact Only):")
    print(stats_df)

    # Perform Z-test
    # Comparison: Defense/Security vs Civilian
    if 'Defense/Security' in stats_df.index and 'Civilian' in stats_df.index:
        n_def = stats_df.loc['Defense/Security', 'Total_Systems']
        x_def = stats_df.loc['Defense/Security', 'Opaque_Systems']
        p_def = stats_df.loc['Defense/Security', 'Opacity_Rate']
        
        n_civ = stats_df.loc['Civilian', 'Total_Systems']
        x_civ = stats_df.loc['Civilian', 'Opaque_Systems']
        p_civ = stats_df.loc['Civilian', 'Opacity_Rate']
        
        # Pooled probability
        p_pool = (x_def + x_civ) / (n_def + n_civ)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_def + 1/n_civ))
        
        if se == 0:
            print("Standard Error is 0, cannot perform Z-test.")
        else:
            z_score = (p_def - p_civ) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            print(f"\nZ-Test Results:")
            print(f"  Defense Opacity: {p_def:.2%}")
            print(f"  Civilian Opacity: {p_civ:.2%}")
            print(f"  Difference: {p_def - p_civ:.2%}")
            print(f"  Z-score: {z_score:.4f}")
            print(f"  P-value: {p_value:.4e}")
            
            alpha = 0.05
            if p_value < alpha:
                print("  Result: Statistically Significant (Reject Null)")
            else:
                print("  Result: Not Significant (Fail to Reject Null)")

        # 7. Visualization: Stacked Bar Chart of Transparency Compliance
        # We will plot 'Opaque' vs 'Transparent' (which is 1 - Opaque Rate)
        
        # Data preparation
        # We want a stacked bar for each agency type.
        # Bottom bar: Opaque Rate
        # Top bar: Transparent Rate
        
        categories = stats_df.index
        opaque_rates = stats_df['Opacity_Rate']
        transparent_rates = 1 - opaque_rates
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot Opaque (Red)
        ax.bar(categories, opaque_rates, label='Opaque (Notice/Opt-out denied)', color='#d62728', alpha=0.8)
        
        # Plot Transparent (Blue) - stacked on top
        ax.bar(categories, transparent_rates, bottom=opaque_rates, label='Transparent/Other', color='#1f77b4', alpha=0.8)
        
        ax.set_ylabel('Proportion of High Impact Systems')
        ax.set_title('Transparency Compliance: Defense vs Civilian Agencies\n(High Impact AI Systems)')
        ax.legend(loc='lower right')
        
        # Add percentage labels
        for i, (cat, op_rate) in enumerate(zip(categories, opaque_rates)):
            # Label for Opaque
            ax.text(i, op_rate / 2, f"{op_rate:.1%}", ha='center', va='center', color='white', fontweight='bold')
            # Label for Transparent
            tr_rate = 1 - op_rate
            ax.text(i, op_rate + tr_rate / 2, f"{tr_rate:.1%}", ha='center', va='center', color='white', fontweight='bold')

        plt.tight_layout()
        plt.show()
        
    else:
        print("\nInsufficient data groups to perform Z-test comparison.")

if __name__ == "__main__":
    run_experiment()