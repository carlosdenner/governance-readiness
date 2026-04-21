import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running locally in a different structure
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- 1. Date Parsing ---
# Function to extract year from various formats
def extract_year(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val)
    # Look for 4 consecutive digits
    match = re.search(r'(\d{4})', val_str)
    if match:
        year = int(match.group(1))
        # Basic sanity check for year range (e.g., 1980 to 2030)
        if 1980 <= year <= 2030:
            return year
    return np.nan

eo_df['impl_year'] = eo_df['20_date_implemented'].apply(extract_year)

# Filter out rows with no valid year
valid_date_df = eo_df.dropna(subset=['impl_year']).copy()

# --- 2. Construct Governance Score ---
# Select key binary/control columns representing governance maturity
gov_cols = [
    '28_iqa_compliance',        # Data Quality
    '40_has_ato',               # Security/Auth
    '52_impact_assessment',     # Impact Assessment
    '55_independent_eval',      # Independent Eval
    '56_monitor_postdeploy',    # Monitoring
    '61_adverse_impact',        # Adverse Impact check
    '62_disparity_mitigation',  # Bias Mitigation
    '65_appeal_process'         # Human Recourse
]

# Normalize to 0/1. strict check for 'Yes' case-insensitive
# Note: Some fields are verbose. We assume containing 'yes' implies presence of control.
for col in gov_cols:
    if col in valid_date_df.columns:
        valid_date_df[f'score_{col}'] = valid_date_df[col].astype(str).str.contains('yes', case=False, na=False).astype(int)
    else:
        valid_date_df[f'score_{col}'] = 0

score_cols = [f'score_{c}' for c in gov_cols]
valid_date_df['gov_score_raw'] = valid_date_df[score_cols].sum(axis=1)
valid_date_df['gov_score_pct'] = (valid_date_df['gov_score_raw'] / len(gov_cols)) * 100

# --- 3. Binning ---
# Legacy: < 2021 (Pre-2021)
# Modern: >= 2021 (2021 and later)
valid_date_df['cohort'] = valid_date_df['impl_year'].apply(lambda x: 'Modern (>=2021)' if x >= 2021 else 'Legacy (<2021)')

legacy_scores = valid_date_df[valid_date_df['cohort'] == 'Legacy (<2021)']['gov_score_pct']
modern_scores = valid_date_df[valid_date_df['cohort'] == 'Modern (>=2021)']['gov_score_pct']

# --- 4. Statistical Test ---
t_stat, p_val = stats.ttest_ind(legacy_scores, modern_scores, equal_var=False, nan_policy='omit')

# --- 5. Output Results ---
print("Analysis of Governance Scores by Implementation Era")
print("---------------------------------------------------")
print(f"Total records with valid dates: {len(valid_date_df)}")
print(f"Legacy Cohort Size (<2021):   {len(legacy_scores)}")
print(f"Modern Cohort Size (>=2021):  {len(modern_scores)}")
print("\nMean Governance Readiness Score (0-100%):")
print(f"  Legacy: {legacy_scores.mean():.2f}%")
print(f"  Modern: {modern_scores.mean():.2f}%")
print(f"\nDifference: {modern_scores.mean() - legacy_scores.mean():.2f}%")
print(f"T-test results: Statistic={t_stat:.4f}, p-value={p_val:.4e}")

if p_val < 0.05:
    print("Result: Statistically Significant Difference.")
else:
    print("Result: No Statistically Significant Difference.")

# --- 6. Visualization ---
plt.figure(figsize=(10, 6))
data_to_plot = [legacy_scores, modern_scores]
labels = [f'Legacy (<2021)\nn={len(legacy_scores)}', f'Modern (>=2021)\nn={len(modern_scores)}']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
plt.title('Governance Readiness Scores: Pre- vs. Post-2021 Deployment')
plt.ylabel('Governance Score (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate with p-value
top_val = max(valid_date_df['gov_score_pct'].max(), 10) + 2
plt.text(1.5, top_val, f'p = {p_val:.4e}', ha='center', va='bottom', fontsize=12, color='red')

plt.tight_layout()
plt.show()
