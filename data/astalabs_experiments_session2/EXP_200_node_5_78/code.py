import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def run_experiment():
    # Load dataset
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        try:
            df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        except FileNotFoundError:
            print("Error: Dataset not found.")
            return

    # Filter for EO 13960 source
    df = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Loaded {len(df)} records from EO 13960.")

    # --- Logic for Public Facing ---
    # 27_public_info: Explicit 'Yes'/'No'.
    def is_public_info(val):
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        return s == 'yes'

    # 26_public_service: Free text descriptions implies True, unless empty or 'No'.
    def is_public_service(val):
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        if s == '' or s == 'no' or s == 'nan':
            return False
        return True

    df['public_facing'] = df['27_public_info'].apply(is_public_info) | df['26_public_service'].apply(is_public_service)

    # --- Logic for AI Notice ---
    # 59_ai_notice: Categorical.
    # Positive keywords: 'online', 'in-person', 'email', 'telephone', 'other'
    # Negative/Neutral: 'n/a', 'none', 'waived', 'not safety'
    def has_notice(val):
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        
        # explicit negatives
        if any(x in s for x in ['n/a', 'none of the above', 'waived', 'not safety']):
            return False
            
        # explicit positives
        if any(x in s for x in ['online', 'in-person', 'email', 'telephone', 'other', 'terms', 'instruction']):
            return True
            
        return False

    df['has_notice'] = df['59_ai_notice'].apply(has_notice)

    # Grouping
    public_group = df[df['public_facing']]
    internal_group = df[~df['public_facing']]

    n_public = len(public_group)
    n_internal = len(internal_group)
    
    print(f"\n--- Categorization ---")
    print(f"Public Facing: {n_public}")
    print(f"Internal/Non-Public: {n_internal}")

    if n_public == 0 or n_internal == 0:
        print("Cannot perform test: One group is empty.")
        return

    # Calculate rates
    n_public_notice = public_group['has_notice'].sum()
    n_internal_notice = internal_group['has_notice'].sum()

    rate_public = n_public_notice / n_public if n_public > 0 else 0
    rate_internal = n_internal_notice / n_internal if n_internal > 0 else 0

    print("\n--- Descriptive Statistics ---")
    print(f"Public-Facing Systems ({n_public}):")
    print(f"  With AI Notice: {n_public_notice} ({rate_public:.2%})")
    print(f"Internal/Other Systems ({n_internal}):")
    print(f"  With AI Notice: {n_internal_notice} ({rate_internal:.2%})")

    # Statistical Test
    observed = np.array([
        [n_public_notice, n_public - n_public_notice],
        [n_internal_notice, n_internal - n_internal_notice]
    ])
    
    # Check for zeroes in rows/cols to avoid error, though chi2_contingency handles 0 observed well, it fails if expected is 0.
    # If row sums are 0, we can't do it.
    if n_public == 0 or n_internal == 0:
         print("Skipping test due to empty group.")
    elif (n_public_notice == 0 and n_internal_notice == 0):
         print("\nResult: No notices found in EITHER group. Rates are identical (0%).")
         print("Hypothesis Status: Technically supported (no difference), but functionally a universal failure of transparency.")
    else:
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        print("\n--- Statistical Test Results (Chi-Square) ---")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4e}")

        alpha = 0.05
        print("\n--- Interpretation ---")
        if p < alpha:
            print("Result: Significant difference detected.")
            if rate_public > rate_internal:
                print("Direction: Public-facing systems are significantly MORE likely to provide notice.")
                print("Hypothesis Status: REJECTED (Transparency works better for public systems).")
            else:
                print("Direction: Public-facing systems are significantly LESS likely to provide notice.")
                print("Hypothesis Status: SUPPORTED (Paradox confirmed).")
        else:
            print("Result: No significant difference detected.")
            print(f"Gap: {(rate_public - rate_internal)*100:.2f} percentage points.")
            print("Hypothesis Status: SUPPORTED (Paradox confirmed - public status does not significantly improve transparency).")

        # Visualization
        plt.figure(figsize=(8, 6))
        categories = ['Public Facing', 'Internal/Non-Public']
        percentages = [rate_public * 100, rate_internal * 100]
        
        bars = plt.bar(categories, percentages, color=['#d62728', '#7f7f7f'], edgecolor='black', alpha=0.8)
        plt.ylabel('Percentage with AI Notice (%)')
        plt.title('Transparency Gap: AI Notice Rates by Deployment Type')
        plt.ylim(0, max(max(percentages)*1.2, 5)) # Ensure at least 0-5 scale
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_experiment()