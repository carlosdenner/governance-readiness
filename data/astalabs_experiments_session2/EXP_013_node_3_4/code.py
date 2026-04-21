import pandas as pd
import scipy.stats as stats
import numpy as np

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
print(f"Loaded {len(atlas_df)} ATLAS cases.")

# 1. Flag Generative AI systems
# Keywords: LLM, GPT, chat, gen, diffusion.
genai_keywords = ['llm', 'gpt', 'chat', 'generative', 'diffusion', 'foundation model', 'transformer', 'bert']

def check_genai(row):
    text = (str(row['name']) + ' ' + str(row['summary'])).lower()
    for kw in genai_keywords:
        if kw in text:
            return True
    return False

atlas_df['is_genai'] = atlas_df.apply(check_genai, axis=1)

# 2. Flag 'Defense Evasion' tactics
# The tactics column contains strings like '{{defense_evasion.id}}'.
# We look for 'defense_evasion' (case insensitive).
def check_defense_evasion(val):
    if pd.isna(val):
        return False
    return 'defense_evasion' in str(val).lower()

atlas_df['has_defense_evasion'] = atlas_df['tactics'].apply(check_defense_evasion)

# 3. Generate Contingency Table
# Rows: GenAI vs Non-GenAI
# Cols: Defense Evasion vs No Defense Evasion
contingency_table = pd.crosstab(atlas_df['is_genai'], atlas_df['has_defense_evasion'])

# Ensure the table is 2x2 even if some categories are missing
# We expect index [False, True] and columns [False, True]
contingency_table = contingency_table.reindex(index=[False, True], columns=[False, True], fill_value=0)

# Rename for clarity
contingency_table.index = ['Non-GenAI', 'GenAI']
contingency_table.columns = ['No Defense Evasion', 'Has Defense Evasion']

print("\nContingency Table (Frequency of Defense Evasion Tactics):")
print(contingency_table)

# 4. Fisher's Exact Test
oddsratio, pvalue = stats.fisher_exact(contingency_table)
print(f"\nFisher's Exact Test p-value: {pvalue:.4f}")
print(f"Odds Ratio: {oddsratio:.4f}")

# Interpretation
alpha = 0.05
if pvalue < alpha:
    print("Result: Statistically significant difference in Defense Evasion prevalence.")
else:
    print("Result: No statistically significant difference found.")
