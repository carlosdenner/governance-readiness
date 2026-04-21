import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess

# function to install packages if needed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Try importing seaborn, install if missing
try:
    import seaborn as sns
except ImportError:
    install('seaborn')
    import seaborn as sns

# Try importing scipy, install if missing
try:
    from scipy.stats import chi2_contingency, fisher_exact
except ImportError:
    install('scipy')
    from scipy.stats import chi2_contingency, fisher_exact

# 1. Load dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for 'eo13960_scored'
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# 3. Data Cleaning & Mapping

# Mapping for Impact Assessment (Strict implementation)
def map_impact(val):
    s = str(val).strip().lower()
    if s == 'yes':
        return 1
    return 0

# Mapping for Stakeholder Consultation (Text analysis)
def map_consult(val):
    s = str(val).strip().lower()
    # Handle missing/nan
    if s == 'nan' or s == '':
        return 0
    # explicit negatives
    if 'none' in s:
        return 0
    if 'n/a' in s:
        return 0
    if 'waived' in s:
        return 0
    # default to positive if it contains content that isn't negative
    return 1

col_impact = '52_impact_assessment'
col_consult = '63_stakeholder_consult'

eo_df['impact_bin'] = eo_df[col_impact].apply(map_impact)
eo_df['consult_bin'] = eo_df[col_consult].apply(map_consult)

# 4. Construct Contingency Table
contingency = pd.crosstab(eo_df['impact_bin'], eo_df['consult_bin'])
contingency.index.name = "Impact Assessment"
contingency.columns.name = "Stakeholder Consult"

# Ensure 2x2
contingency = contingency.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

print("\nContingency Table:")
print(contingency)

# Check for zero variance
if (contingency.sum(axis=1) == 0).any() or (contingency.sum(axis=0) == 0).any():
    print("\nWarning: One of the variables has zero variance. Cannot compute meaningful statistics.")
    # Create a dummy result for the sake of flow, or exit
    phi = 0
    odds_ratio = 0
    p = 1.0
else:
    # 5. Statistical Tests
    # Use Chi2
    chi2, p, dof, ex = chi2_contingency(contingency)
    
    # Calculate Phi Coefficient
    a = contingency.loc[0, 0]
    b = contingency.loc[0, 1]
    c = contingency.loc[1, 0]
    d = contingency.loc[1, 1]
    
    phi_denom = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    phi = (a*d - b*c) / phi_denom if phi_denom > 0 else 0
    
    # Odds Ratio
    if b*c == 0:
        odds_ratio = np.inf
    else:
        odds_ratio = (d * a) / (b * c)

    print(f"\nStatistical Results:")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")
    print(f"Phi Coefficient: {phi:.4f}")
    print(f"Odds Ratio: {odds_ratio:.4f}")

    # Interpretation
    if p < 0.05 and phi > 0.3:
        print("\nResult: Hypothesis CONFIRMED. Significant positive clustering observed.")
    elif p < 0.05 and phi > 0:
        print("\nResult: Hypothesis WEAK. Significant but low correlation.")
    else:
        print("\nResult: Hypothesis REJECTED.")

# 6. Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No/Other', 'Yes'], yticklabels=['No/Other', 'Yes'])
plt.title(f'Competency Bundle: Impact Assessment vs Stakeholder Consult\n(Phi={phi:.2f}, OR={odds_ratio:.2f})')
plt.xlabel('Stakeholder Consultation')
plt.ylabel('Impact Assessment')
plt.tight_layout()
plt.show()