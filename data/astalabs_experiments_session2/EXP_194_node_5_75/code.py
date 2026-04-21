import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re
import os

# Attempt to load the dataset from the parent directory as instructed, falling back to current.
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset 'astalabs_discovery_all_data.csv' not found in ../ or ./.")
        exit(1)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()

# Keywords to identify Generative AI systems
genai_keywords = [
    'llm', 'gpt', 'generative', 'diffusion', 'chatbot', 'chat', 
    'transformer', 'bert', 'dall-e', 'midjourney', 'stable diffusion', 
    'openai', 'anthropic', 'bard', 'bing', 'copilot', 'gemini', 'llama',
    'genai'
]

def classify_genai(row):
    # Combine name and summary for keyword search
    name = str(row['name']) if pd.notna(row['name']) else ''
    summary = str(row['summary']) if pd.notna(row['summary']) else ''
    text = (name + " " + summary).lower()
    return any(kw in text for kw in genai_keywords)

# Apply classification
atlas_df['is_genai'] = atlas_df.apply(classify_genai, axis=1)

# Function to parse and count techniques
def count_techniques(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    # Handle potential string formats (pipe, comma, or stringified list)
    val_str = str(val)
    # Basic cleanup for stringified lists if present
    val_str = val_str.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
    # Split by common delimiters
    parts = re.split(r'[|;,\n]', val_str)
    # Count unique non-empty items
    unique_techniques = {p.strip() for p in parts if p.strip()}
    return len(unique_techniques)

# Calculate technique counts
atlas_df['technique_count'] = atlas_df['techniques'].apply(count_techniques)

# Split into groups
genai_group = atlas_df[atlas_df['is_genai']]['technique_count']
trad_group = atlas_df[~atlas_df['is_genai']]['technique_count']

# Output Statistics
print("--- Analysis of Attack Sophistication (Technique Count) ---")
print(f"Total Cases Analyzed: {len(atlas_df)}")
print(f"Generative AI Cases: {len(genai_group)}")
print(f"Traditional AI Cases: {len(trad_group)}")

print("\n--- Descriptive Statistics ---")
print(f"GenAI:       Mean = {genai_group.mean():.2f}, Std = {genai_group.std():.2f}, Median = {genai_group.median()}")
print(f"Traditional: Mean = {trad_group.mean():.2f}, Std = {trad_group.std():.2f}, Median = {trad_group.median()}")

# Perform Welch's t-test (equal_var=False)
t_stat, p_val = stats.ttest_ind(genai_group, trad_group, equal_var=False)

print("\n--- Two-Sample T-Test Results ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value:     {p_val:.4f}")

if p_val < 0.05:
    print("Conclusion: Statistically significant difference in technique counts.")
else:
    print("Conclusion: No statistically significant difference in technique counts.")

# Visualization: Violin Plot
plt.figure(figsize=(8, 6))
dataset = [genai_group, trad_group]
labels = ['GenAI', 'Traditional AI']

parts = plt.violinplot(dataset, showmeans=True, showmedians=True)

# Customize plot
plt.xticks([1, 2], labels)
plt.ylabel('Number of Unique Techniques per Case')
plt.title('Attack Complexity: GenAI vs Traditional AI')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Color styling
colors = ['#FF9999', '#66B2FF']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i % len(colors)])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

plt.tight_layout()
plt.show()