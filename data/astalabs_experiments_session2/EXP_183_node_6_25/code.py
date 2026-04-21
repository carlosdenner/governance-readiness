import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# [debug] Only used to ensure valid execution context
# print("Starting experiment...")

# 1. Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running in a different environment structure
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# 3. Parse Date and Create Periods
# The 'date' column is expected to be in a parseable format (e.g., YYYY-MM-DD)
aiid_df['date_parsed'] = pd.to_datetime(aiid_df['date'], errors='coerce')

# Drop rows where date could not be parsed
aiid_df = aiid_df.dropna(subset=['date_parsed'])

# Define cut-off date: 2023-01-01
cutoff_date = pd.to_datetime('2023-01-01')

# Create Period Column
aiid_df['period'] = aiid_df['date_parsed'].apply(
    lambda d: 'Post-2022' if d >= cutoff_date else 'Pre-2023'
)

# 4. Classify Generative AI Incidents
# We will search for keywords in '84: Known AI Technology', 'title', and 'description' if available.
# Note: Column names in the CSV often have prefixes like "84: ". We handle this dynamically.

# Identify relevant columns
cols_to_search = []
possible_tech_cols = [c for c in aiid_df.columns if 'Known AI Technology' in c]
possible_desc_cols = [c for c in aiid_df.columns if 'description' in c.lower() or 'summary' in c.lower() or 'title' in c.lower()]

cols_to_search.extend(possible_tech_cols)
cols_to_search.extend(possible_desc_cols)

# Define GenAI keywords
genai_keywords = [
    'generative', 'llm', 'large language model', 'gpt', 'chatbot', 
    'transformer', 'diffusion', 'dalle', 'midjourney', 'stable diffusion', 
    'bard', 'chatgpt', 'llama', 'copilot', 'gemini', 'anthropic', 'claude',
    'foundation model', 'text-to-image', 'genai'
]

def check_genai(row):
    text_blob = ""
    for col in cols_to_search:
        val = row[col]
        if pd.notna(val):
            text_blob += str(val).lower() + " "
    
    for kw in genai_keywords:
        if kw in text_blob:
            return True
    return False

aiid_df['is_genai'] = aiid_df.apply(check_genai, axis=1)

# 5. Summary Statistics and Contingency Table
contingency_table = pd.crosstab(aiid_df['period'], aiid_df['is_genai'])
contingency_table.columns = ['Non-GenAI', 'GenAI']

print("--- Contingency Table: Period vs. Generative AI ---")
print(contingency_table)
print("\n")

# Calculate Proportions
summary = contingency_table.copy()
summary['Total'] = summary['Non-GenAI'] + summary['GenAI']
summary['GenAI_Rate'] = summary['GenAI'] / summary['Total']

print("--- Proportions ---")
print(summary)
print("\n")

# 6. Statistical Test (Chi-Square)
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"--- Chi-Square Test Results ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Degrees of Freedom: {dof}")

if p < 0.05:
    print("Result: Statistically Significant Shift detected.")
else:
    print("Result: No Statistically Significant Shift detected.")

# Calculate Fold Increase (Relative Risk)
try:
    rate_pre = summary.loc['Pre-2023', 'GenAI_Rate']
    rate_post = summary.loc['Post-2022', 'GenAI_Rate']
    fold_increase = rate_post / rate_pre if rate_pre > 0 else np.nan
    print(f"Fold Increase (Post / Pre): {fold_increase:.2f}x")
except KeyError:
    print("Could not calculate fold increase due to missing periods.")

# 7. Visualization
plt.figure(figsize=(8, 6))
periods = summary.index
rates = summary['GenAI_Rate'] * 100

bars = plt.bar(periods, rates, color=['skyblue', 'salmon'])
plt.ylabel('Percentage of Incidents involving GenAI (%)')
plt.title('Prevalence of Generative AI in Incidents (Pre-2023 vs Post-2022)')
plt.ylim(0, max(rates) * 1.2)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()
