import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import re

# [debug] Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# --- Feature Engineering: Technology Type ---
# Define keywords for Generative AI
genai_keywords = [
    r'generative', r'llm', r'large language model', r'gpt', r'chatgpt', r'chatbot',
    r'diffusion', r'dall-e', r'midjourney', r'stable diffusion', r'transformer',
    r'bert', r'gan', r'deepfake', r'synthetic', r'text-to-image', r'text-to-video',
    r'bard', r'llama', r'claude', r'copilot', r'gemini'
]

def classify_tech(row):
    # Combine relevant columns for search
    text = str(row.get('Known AI Technology', '')) + " " + str(row.get('Potential AI Technology', '')) + " " + str(row.get('description', ''))
    text = text.lower()
    
    for keyword in genai_keywords:
        if re.search(keyword, text):
            return 'Generative AI'
    return 'Discriminative/Predictive AI'

aiid_df['Technology_Type'] = aiid_df.apply(classify_tech, axis=1)

# --- Feature Engineering: Reputational Harm ---
# Define keywords for Reputational Harm
reputation_keywords = [
    r'reputation', r'reputational', r'defamation', r'libel', r'slander',
    r'brand damage', r'public relation', r'scandal', r'embarrassment', r'discredit'
]

def classify_harm(row):
    # Combine relevant columns for search
    text = str(row.get('Harm Domain', '')) + " " + str(row.get('Tangible Harm', '')) + " " + str(row.get('Special Interest Intangible Harm', ''))
    text = text.lower()
    
    for keyword in reputation_keywords:
        if re.search(keyword, text):
            return 'Reputational'
    return 'Other Harm'

aiid_df['Harm_Class'] = aiid_df.apply(classify_harm, axis=1)

# --- Analysis ---
print(f"Total Incidents Analysis: {len(aiid_df)}")
print("\nDistribution of Technology Type:")
print(aiid_df['Technology_Type'].value_counts())
print("\nDistribution of Harm Class:")
print(aiid_df['Harm_Class'].value_counts())

# Contingency Table
contingency_table = pd.crosstab(aiid_df['Technology_Type'], aiid_df['Harm_Class'])
print("\nContingency Table (Technology vs. Harm):")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n--- Chi-Square Test Results ---")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Degrees of Freedom: {dof}")

# Calculate percentages for interpretation
# Row-wise normalization (Probability of Harm given Tech)
row_props = pd.crosstab(aiid_df['Technology_Type'], aiid_df['Harm_Class'], normalize='index') * 100
print("\nRow Percentages (Propensity for Harm Type by Tech):")
print(row_props)

# --- Visualization ---
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap: AI Technology Type vs. Reputational Harm')
plt.xlabel('Harm Category')
plt.ylabel('Technology Type')
plt.tight_layout()
plt.show()

# Stacked Bar Chart for Proportions
ax = row_props.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title('Proportion of Reputational vs. Other Harm by Technology Type')
plt.ylabel('Percentage')
plt.xlabel('Technology Type')
plt.legend(title='Harm Class', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add labels
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center')

plt.tight_layout()
plt.show()