import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 subset
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Step 1: Define Audience (Public vs Internal) ---
# Logic: '26_public_service' contains descriptions for public services.
# We assume populated (and meaningful) text implies 'Public-Facing'.
# Null or empty/short artifacts implies 'Internal'.

def categorize_audience(val):
    if pd.isna(val):
        return 'Internal'
    s = str(val).strip()
    if len(s) < 3:  # Filter out artifacts like '.', ' ', or '\n'
        return 'Internal'
    return 'Public-Facing'

eo_df['audience_clean'] = eo_df['26_public_service'].apply(categorize_audience)

# --- Step 2: Define Opt-Out (Yes vs No) ---
# Logic: '67_opt_out' contains 'Yes', verbose 'No...', or 'Waived'.

def categorize_opt_out(val):
    if pd.isna(val):
        return None
    s = str(val).lower().strip()
    
    if s == 'yes' or s.startswith('yes'):
        return 'Yes'
    if s.startswith('no') or 'waived' in s:
        return 'No'
    return None # Exclude 'Other', 'N/A' if ambiguous, or NaN

eo_df['opt_out_clean'] = eo_df['67_opt_out'].apply(categorize_opt_out)

# --- Step 3: Filter Valid Data ---
# We only analyze rows where a definitive Opt-Out status is recorded.
analysis_df = eo_df.dropna(subset=['opt_out_clean']).copy()

print("Data Preparation Complete.")
print(f"Total EO 13960 Rows: {len(eo_df)}")
print(f"Rows with Valid Opt-Out Status: {len(analysis_df)}")
print("\nDistribution of Audience in Analysis Set:")
print(analysis_df['audience_clean'].value_counts())

# --- Step 4: Statistical Analysis ---
if len(analysis_df) > 0:
    # Contingency Table
    crosstab = pd.crosstab(analysis_df['audience_clean'], analysis_df['opt_out_clean'])
    print("\nContingency Table (Count):")
    print(crosstab)
    
    # Check if we have enough data dimensions
    if crosstab.shape == (2, 2):
        # Percentages
        crosstab_pct = pd.crosstab(analysis_df['audience_clean'], analysis_df['opt_out_clean'], normalize='index') * 100
        print("\nOpt-Out Availability Rates (%):")
        print(crosstab_pct)
        
        # Chi-Square
        chi2, p, dof, ex = chi2_contingency(crosstab)
        print(f"\nChi-Square Test Results:")
        print(f"Statistic: {chi2:.4f}")
        print(f"P-Value: {p:.4e}")
        
        # Interpretation
        if p < 0.05:
            print("Result: Statistically Significant Association.")
        else:
            print("Result: No Significant Association.")
            
        # Visualization
        plt.figure(figsize=(8, 6))
        yes_rates = crosstab_pct['Yes']
        colors = ['#2ca02c', '#1f77b4'] # Green vs Blue
        ax = yes_rates.plot(kind='bar', color=colors, alpha=0.9, edgecolor='black', rot=0)
        
        plt.title("The 'Forced Participation' Paradox:\nOpt-Out Availability by Audience")
        plt.ylabel("Use Cases with Opt-Out (%)")
        plt.xlabel("Target Audience")
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        for i, v in enumerate(yes_rates):
            ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
            
        plt.tight_layout()
        plt.show()
        
    else:
        print("\nInsufficient dimensions for Chi-Square (need 2x2).")
        # Fallback print if one category is missing (e.g., all Internal are No)
        print("Raw Counts:")
        print(crosstab)
else:
    print("No valid data available for analysis.")
