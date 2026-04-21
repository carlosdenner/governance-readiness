import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    # Using low_memory=False to avoid mixed type warnings
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for relevant source table
subset = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Initial subset size: {len(subset)}")

# --- 1. Variable Mapping ---
# Note: '26_public_service' was found to be extremely sparse (93% missing) and unreliable 
# (classifying 'CBP One' as missing/internal). 
# We substitute '27_public_info' ('Has the agency published information...') as a proxy 
# for public-facing transparency vs internal/obscure systems.

def normalize_binary(val):
    s = str(val).strip().lower()
    if s == 'yes':
        return True
    elif s == 'no':
        return False
    else:
        return None

subset['is_public'] = subset['27_public_info'].apply(normalize_binary)

# Outcome: Appeal Process
# We treat 'Yes' as True, and everything else (No, NaN, N/A) as False/No Appeal.
subset['has_appeal'] = subset['65_appeal_process'].apply(lambda x: str(x).strip().lower() == 'yes')

# Drop rows where our independent variable (Deployment Type) is undefined
clean_subset = subset.dropna(subset=['is_public'])

print(f"Cleaned subset size (valid Public/Internal label): {len(clean_subset)}")
print("Group sizes:")
print(clean_subset['is_public'].value_counts())

# --- 2. Analysis ---

# Contingency Table
contingency_table = pd.crosstab(clean_subset['is_public'], clean_subset['has_appeal'])

# Ensure we have the right shape before labeling
if contingency_table.shape[0] == 2:
    contingency_table.index = ['Internal-Facing', 'Public-Facing']
    contingency_table.columns = ['No Appeal Process', 'Has Appeal Process']
else:
    print("Warning: Contingency table does not have 2 rows. Check data.")
    print(contingency_table)

print("\nContingency Table:")
print(contingency_table)

# Proportions
public_group = clean_subset[clean_subset['is_public'] == True]
internal_group = clean_subset[clean_subset['is_public'] == False]

public_appeal_rate = public_group['has_appeal'].mean()
internal_appeal_rate = internal_group['has_appeal'].mean()

print(f"\nAppeal Process Availability Rate - Public-Facing (n={len(public_group)}): {public_appeal_rate:.2%}")
print(f"Appeal Process Availability Rate - Internal-Facing (n={len(internal_group)}): {internal_appeal_rate:.2%}")

# Statistical Test (Chi-Square)
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# --- 3. Visualization ---

labels = ['Internal-Facing', 'Public-Facing']
rates = [internal_appeal_rate, public_appeal_rate]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, rates, color=['#A9A9A9', '#4682B4'])
plt.ylabel('Proportion with Appeal Process')
plt.title('Appeal Process Availability by Deployment Type')
plt.ylim(0, max(rates) * 1.3 if max(rates) > 0 else 0.1)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1%}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()