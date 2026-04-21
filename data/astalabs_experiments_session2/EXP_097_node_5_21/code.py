import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# 3. Feature Engineering: Sector
def classify_sector(sector_str):
    if pd.isna(sector_str):
        return None
    
    s = sector_str.lower()
    
    # Public Sector keywords
    public_keywords = ['public administration', 'defense', 'law enforcement', 'education', 'social work', 'government']
    if any(k in s for k in public_keywords):
        return 'Public'
    
    # Private Sector keywords (if not public)
    private_keywords = ['financial', 'manufacturing', 'retail', 'entertainment', 'transportation', 
                        'accommodation', 'information', 'communication', 'professional', 'real estate', 'arts']
    if any(k in s for k in private_keywords):
        return 'Private'
    
    return None

aiid_df['Sector_Group'] = aiid_df['Sector of Deployment'].apply(classify_sector)

# 4. Feature Engineering: Harm
def classify_harm(harm_str):
    if pd.isna(harm_str) or harm_str == 'unclear':
        return None
    
    if harm_str == 'tangible harm definitively occurred':
        return 'Tangible'
    elif harm_str in ['no tangible harm, near-miss, or issue', 
                      'imminent risk of tangible harm (near miss) did occur', 
                      'non-imminent risk of tangible harm (an issue) occurred']:
        return 'Intangible/Risk'
    
    return None

aiid_df['Harm_Category'] = aiid_df['Tangible Harm'].apply(classify_harm)

# 5. Drop rows with missing values in relevant columns
analysis_df = aiid_df.dropna(subset=['Sector_Group', 'Harm_Category'])

# 6. Create Contingency Table
contingency_table = pd.crosstab(analysis_df['Sector_Group'], analysis_df['Harm_Category'])

print("--- Contingency Table (Sector vs. Harm) ---")
print(contingency_table)

# 7. Statistical Tests
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")

# 8. Odds Ratio Calculation
# OR = (a*d) / (b*c)
# Table structure usually: 
#           Intangible  Tangible
# Private       a          b
# Public        c          d
# We want odds of Intangible in Public vs Private, or Tangible in Private vs Public.
# Let's calculate Odds of Tangible Harm for Private vs. Public.

if 'Tangible' in contingency_table.columns and 'Private' in contingency_table.index:
    # Counts
    private_tangible = contingency_table.loc['Private', 'Tangible']
    private_intangible = contingency_table.loc['Private', 'Intangible/Risk']
    public_tangible = contingency_table.loc['Public', 'Tangible']
    public_intangible = contingency_table.loc['Public', 'Intangible/Risk']
    
    # Odds
    odds_private = private_tangible / private_intangible if private_intangible > 0 else np.nan
    odds_public = public_tangible / public_intangible if public_intangible > 0 else np.nan
    
    odds_ratio = odds_private / odds_public if odds_public > 0 else np.nan
    
    print(f"\nOdds of Tangible Harm (Private): {odds_private:.4f}")
    print(f"Odds of Tangible Harm (Public): {odds_public:.4f}")
    print(f"Odds Ratio (Private vs Public for Tangible Harm): {odds_ratio:.4f}")
else:
    print("\nCould not calculate Odds Ratio due to missing categories in table.")

# 9. Visualization
contingency_table.plot(kind='bar', stacked=True)
plt.title('Harm Category Distribution by Sector')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()