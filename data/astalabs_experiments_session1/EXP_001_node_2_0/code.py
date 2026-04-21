import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os

# Define file path
filename = 'step3_incident_coding.csv'

# Attempt to load the dataset
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    # Fallback to parent directory if not found in current
    df = pd.read_csv(os.path.join('..', filename))

# Define the mapping for Harm Types
security_harms = ['security', 'supply_chain', 'intellectual_property', 'autonomy_misuse']
non_security_harms = ['privacy', 'bias_discrimination', 'reliability']

def categorize_harm(harm):
    h = str(harm).strip()
    if h in security_harms:
        return 'Security'
    elif h in non_security_harms:
        return 'Non-Security'
    else:
        return 'Other'

# Apply categorization
df['harm_category'] = df['harm_type'].apply(categorize_harm)

# Filter out 'Other' if any (though metadata suggests all are covered)
df = df[df['harm_category'] != 'Other']

# Generate Contingency Table
# Columns expected in 'trust_integration_split': 'trust-dominant', 'integration-dominant', 'both'
contingency = pd.crosstab(df['harm_category'], df['trust_integration_split'])

print("=== Contingency Table (Harm Category vs. Readiness Split) ===")
print(contingency)

# Perform Chi-Square Test of Independence
stat, p, dof, expected = chi2_contingency(contingency)

print(f"\n=== Statistical Test Results (Chi-Square) ===")
print(f"Chi2 Statistic: {stat:.4f}")
print(f"P-Value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")

# Row Percentages for clearer interpretation
contingency_pct = pd.crosstab(df['harm_category'], df['trust_integration_split'], normalize='index') * 100
print("\n=== Row Percentages ===")
print(contingency_pct.round(2))

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Incident Count'})
plt.title('Heatmap: Harm Category vs. Trust/Integration Split')
plt.xlabel('Competency Split')
plt.ylabel('Harm Category')
plt.tight_layout()
plt.show()