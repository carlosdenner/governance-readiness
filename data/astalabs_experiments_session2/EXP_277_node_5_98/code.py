import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    with np.errstate(divide='ignore', invalid='ignore'):
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        if min((kcorr-1), (rcorr-1)) == 0:
             return 0.0
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

print("Starting analysis...")

# 1. Load Data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"Loaded {len(aiid)} AIID incidents.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# 2. Clean Autonomy Level
autonomy_map = {
    'Autonomy1': 'Low',
    'Autonomy2': 'Medium',
    'Autonomy3': 'High'
}
aiid['Autonomy_Clean'] = aiid['Autonomy Level'].map(autonomy_map)

# Drop rows with unknown autonomy
aiid_clean = aiid.dropna(subset=['Autonomy_Clean']).copy()
print(f"Rows after cleaning Autonomy: {len(aiid_clean)}")

# 3. Derive Harm Type from Text (Title + Description)
# Fill NaNs with empty string
aiid_clean['text_corpus'] = aiid_clean['title'].fillna('') + " " + aiid_clean['description'].fillna('')
aiid_clean['text_corpus'] = aiid_clean['text_corpus'].str.lower()

# Define keywords for Physical Harm
physical_keywords = [
    'death', 'dead', 'die', 'kill', 'fatal', 'mortality', 
    'injury', 'injured', 'hurt', 'wound', 'harm', 
    'crash', 'accident', 'collision', 'hit', 'struck',
    'safety', 'physical', 'violence', 'assault', 'attack',
    'medical', 'health', 'patient', 'hospital', 'surgery'
]

# Regex pattern: word boundaries to avoid partial matches (e.g. 'timeline' matching 'die' if not careful, though 'die' is short)
# specific check: using word boundaries for short words
pattern = '|'.join([f'\\b{w}\\b' for w in physical_keywords])

def categorize_harm(text):
    if pd.isna(text) or text.strip() == '':
        return 'Non-Physical' # Default if no info
    # Check for keywords
    if pd.Series(text).str.contains(pattern, regex=True).any():
        return 'Physical'
    return 'Non-Physical'

# Vectorized apply is faster for regex
aiid_clean['Harm_Type'] = np.where(aiid_clean['text_corpus'].str.contains(pattern, regex=True), 'Physical', 'Non-Physical')

# 4. Analysis
# Contingency Table
contingency_table = pd.crosstab(aiid_clean['Autonomy_Clean'], aiid_clean['Harm_Type'])

# Reorder for ordinality
order = ['Low', 'Medium', 'High']
contingency_table = contingency_table.reindex(order)

print("\n--- Contingency Table: Autonomy Level vs. Derived Harm Type ---")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
v = cramers_v(contingency_table)

print("\n--- Statistical Results ---")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Cramer's V: {v:.4f}")

if p < 0.05:
    print("Conclusion: Statistically significant association found.")
else:
    print("Conclusion: No statistically significant association found.")

# Calculate Proportions for better interpretation
props = contingency_table.div(contingency_table.sum(axis=1), axis=0)
print("\n--- Proportions of Physical Harm by Autonomy ---")
print(props)

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Heatmap of AI Autonomy Level vs. Physical Harm')
plt.ylabel('Autonomy Level')
plt.xlabel('Harm Type')
plt.show()