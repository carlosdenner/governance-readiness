import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

# Load dataset
# Using the provided relative path based on previous successful executions
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored subset
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Step 1: Inspect and Clean Risk Levels (17_impact_type) ---
# Actual values observed: 'Neither', 'Both', 'Rights-Impacting', 'Safety-impacting', 'Safety-Impacting'

def categorize_risk(val):
    if pd.isna(val):
        return np.nan
    val_clean = str(val).strip()
    # High Risk categories: Rights, Safety, or Both
    if val_clean in ['Both', 'Rights-Impacting', 'Safety-impacting', 'Safety-Impacting']:
        return 'High Risk'
    # Standard Risk categories: Neither
    elif val_clean == 'Neither':
        return 'Standard Risk'
    else:
        return np.nan

eo_df['risk_tier'] = eo_df['17_impact_type'].apply(categorize_risk)

# --- Step 2: Inspect and Clean Control Evidence (55_independent_eval) ---
# Actual values include: 'Yes – by the CAIO', 'Planned or in-progress', 'TRUE', etc.

def parse_eval(val):
    if pd.isna(val):
        return 0
    val_lower = str(val).lower()
    # Strict 'Yes' criteria: must start with 'yes' or be explicitly 'true'
    # 'Planned or in-progress' is typically considered NOT yet fully compliant in strict audits,
    # but we will check strict 'Yes' first. 
    if val_lower.startswith('yes') or val_lower == 'true':
        return 1
    return 0

eo_df['has_eval'] = eo_df['55_independent_eval'].apply(parse_eval)

# Drop rows where risk tier is undefined
analysis_df = eo_df.dropna(subset=['risk_tier'])

# --- Step 3: Calculate Statistics ---
groups = analysis_df.groupby('risk_tier')['has_eval'].agg(['sum', 'count', 'mean'])
groups.columns = ['eval_count', 'total', 'proportion']

print("\n--- Descriptive Statistics ---")
print(groups)

# --- Step 4: Statistical Testing ---
if 'High Risk' in groups.index and 'Standard Risk' in groups.index:
    # Counts of successes (evaluations)
    count = np.array([groups.loc['High Risk', 'eval_count'], groups.loc['Standard Risk', 'eval_count']])
    # Total observations
    nobs = np.array([groups.loc['High Risk', 'total'], groups.loc['Standard Risk', 'total']])
    
    # H0: p_high <= p_standard
    # H1: p_high > p_standard (expecting High Risk to have higher eval rates)
    stat, pval = proportions_ztest(count, nobs, alternative='larger')
    
    print(f"\nZ-test Statistic: {stat:.4f}")
    print(f"P-value (one-sided): {pval:.4e}")
    
    alpha = 0.05
    if pval < alpha:
        print("Result: REJECT Null Hypothesis. High Risk systems DO have statistically higher evaluation rates.")
        print("Interpretation: Governance is scaling with risk (Risk-Proportionate).")
    else:
        print("Result: FAIL TO REJECT Null Hypothesis. High Risk systems do NOT have higher evaluation rates.")
        print("Interpretation: Governance is NOT scaling with risk (Risk-Control Mismatch supported).")

    # --- Step 5: Visualization ---
    plt.figure(figsize=(10, 6))
    bar_colors = ['#d9534f' if idx == 'High Risk' else '#5bc0de' for idx in groups.index]
    bars = plt.bar(groups.index, groups['proportion'], color=bar_colors, alpha=0.8)
    
    plt.title('Independent Evaluation Rates by Impact Type (Risk Tier)')
    plt.ylabel('Proportion with Independent Evaluation')
    plt.ylim(0, max(groups['proportion']) * 1.3)  # Add headroom for text
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, 
                 f'{height:.1%}\n(n={int(groups.loc[groups.index[list(bars).index(bar)], "total"])})', 
                 ha='center', va='bottom')
    
    # Add significance annotation
    sig_text = "Significant" if pval < 0.05 else "Not Significant"
    plt.text(0.5, 0.9, f'p-value: {pval:.4e}\n({sig_text})', 
             transform=plt.gca().transAxes, ha='center', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
             
    plt.tight_layout()
    plt.show()

else:
    print("\nError: Could not identify both High Risk and Standard Risk groups in the data.")