import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded {len(aiid)} AIID incidents.")

# Dynamic Column Search
all_cols = aiid.columns.tolist()

# Find Technology columns
tech_candidates = [c for c in all_cols if 'technology' in c.lower()]
print(f"Found Technology columns: {tech_candidates}")

# Find Text columns (Description/Summary)
text_candidates = [c for c in all_cols if 'description' in c.lower() or 'summary' in c.lower()]
print(f"Found Text columns: {text_candidates}")

# Combine technology columns
if not tech_candidates:
    print("Error: No technology columns found. checking for 'AI' in columns.")
    tech_candidates = [c for c in all_cols if 'ai' in c.lower() and 'known' in c.lower()]
    print(f"Alternative Tech columns: {tech_candidates}")

aiid['tech_combined'] = aiid[tech_candidates].apply(lambda x: ' '.join(x.dropna().astype(str).str.lower()), axis=1)

# Define Generative AI keywords
genai_keywords = ['generative', 'llm', 'diffusion', 'chat', 'gpt', 'transformer', 'language model', 'genai', 'chatbot', 'foundation model', 'midjourney', 'dall-e', 'stable diffusion', 'bert', 'large language model']

def classify_tech(text):
    if any(k in text for k in genai_keywords):
        return 'Generative'
    return 'Discriminative'

aiid['tech_type'] = aiid['tech_combined'].apply(classify_tech)

# Combine text columns for keyword search
# We prioritize description/summary columns, but if none, we might look at 'title' or just fall back to empty
if text_candidates:
    aiid['text_combined'] = aiid[text_candidates].apply(lambda x: ' '.join(x.dropna().astype(str).str.lower()), axis=1)
else:
    print("Warning: No description/summary found. Trying 'title'.")
    if 'title' in all_cols:
        aiid['text_combined'] = aiid['title'].astype(str).str.lower()
    else:
        aiid['text_combined'] = ""

# Define ATLAS/Adversarial keywords
adversarial_keywords = [
    'injection', 'jailbreak', 'extraction', 'poisoning', 'evasion', 
    'adversarial', 'prompt', 'red team', 'bypass', 'attack', 'manipulat', 
    'inference', 'inversion', 'membership inference', 'model stealing', 'trojan', 'backdoor'
]

def check_adversarial(text):
    return any(k in text for k in adversarial_keywords)

aiid['has_adversarial_keywords'] = aiid['text_combined'].apply(check_adversarial)

# Generate stats
contingency_table = pd.crosstab(aiid['tech_type'], aiid['has_adversarial_keywords'])
print("\nContingency Table (Rows: Tech Type, Cols: Has Adversarial Keywords):")
print(contingency_table)

# Calculate proportions
summary = aiid.groupby('tech_type')['has_adversarial_keywords'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total Incidents', 'Adversarial Matches', 'Proportion']
print("\nSummary Statistics:")
print(summary)

# Chi-Square Test
chi2, p, dof, ex = chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Visualization
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e']

if not summary.empty:
    prop_plot = summary['Proportion'].plot(kind='bar', color=colors, alpha=0.8)
    plt.title('Proportion of Incidents with Adversarial Keywords by AI Type')
    plt.ylabel('Proportion (0-1)')
    plt.xlabel('AI Technology Type')
    plt.xticks(rotation=0)
    # Set ylim with margin
    top_val = summary['Proportion'].max()
    if top_val > 0:
        plt.ylim(0, top_val * 1.2)
    
    for i, v in enumerate(summary['Proportion']):
        plt.text(i, v + (top_val*0.01), f"{v:.1%}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()
else:
    print("Summary is empty, cannot plot.")
