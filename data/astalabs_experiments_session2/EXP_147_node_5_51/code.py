import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# [debug]
print("Starting experiment: GenAI vs Discriminative AI Harm Profiles (Attempt 3)")

# 1. Load Data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents found: {len(aiid)}")

# 2. Inspect potential text columns for keyword extraction
print("\n--- Column Inspection ---")
potential_text_cols = ['title', 'description', 'summary', 'reports', 'Special Interest Intangible Harm']
for col in potential_text_cols:
    if col in aiid.columns:
        print(f"Column '{col}' sample:", aiid[col].dropna().unique()[:3])
    else:
        print(f"Column '{col}' not found.")

# 3. Tag AI Type (Generative vs Discriminative)
genai_keywords = [
    'generative', 'llm', 'gpt', 'chat', 'chatbot', 'diffusion', 'text-to-image', 
    'image generator', 'deepfake', 'voice cloning', 'gan', 'language model', 
    'transformer', 'midjourney', 'dall-e', 'stable diffusion', 'bard', 'bing chat', 
    'copilot', 'llama', 'mistral', 'claude', 'gemini'
]

def tag_ai_type(row):
    text = " ".join([str(row.get(c, '')) for c in ['Known AI Technology', 'Potential AI Technology', 'title', 'description', 'summary']]).lower()
    if any(k in text for k in genai_keywords):
        return 'Generative AI'
    return 'Discriminative/Other'

aiid['AI_Type'] = aiid.apply(tag_ai_type, axis=1)
print("\nAI Type Distribution:")
print(aiid['AI_Type'].value_counts())

# 4. Categorize Harms using keyword extraction on Text Fields
# Since structured columns failed, we mine 'title', 'description' (if exists), 'reports', and 'Special Interest Intangible Harm'

def map_harm_category(row):
    # Aggregate text from relevant columns
    text_sources = [
        row.get('title', ''),
        row.get('description', ''),
        row.get('summary', ''),
        row.get('Special Interest Intangible Harm', ''),
        row.get('reports', '') # Reports might be long, but useful
    ]
    text = " ".join([str(t) for t in text_sources]).lower()
    
    # Harm Keywords
    
    # Group 1: Societal, Psychological, Reputational (Hypothesized for GenAI)
    societal_keywords = [
        'reputation', 'defamation', 'libel', 'slander', 'psychological', 'harassment', 
        'sexual', 'nude', 'pornography', 'bias', 'discriminat', 'racist', 'sexist', 
        'misinformation', 'disinformation', 'fake news', 'propaganda', 'privacy', 
        'surveillance', 'copyright', 'plagiarism', 'offensive', 'hate speech'
    ]
    
    # Group 2: Economic, Allocative, Financial (Hypothesized for Discriminative)
    economic_keywords = [
        'financial', 'money', 'economic', 'employment', 'hiring', 'job', 'termination', 
        'fired', 'credit', 'loan', 'housing', 'tenant', 'fraud', 'scam', 'theft', 
        'market', 'trading', 'price', 'bank', 'insurance'
    ]
    
    # Group 3: Physical Safety (Common baseline)
    physical_keywords = [
        'death', 'kill', 'died', 'fatal', 'injury', 'injured', 'hurt', 'accident', 
        'crash', 'collision', 'medical', 'patient', 'hospital', 'health', 'physical safety'
    ]
    
    # Priority: Check specifically for the distinct categories first
    # We map to the dominant category found. If multiple, we prioritize based on hypothesis relevance or hierarchy.
    # Let's check existence.
    has_societal = any(k in text for k in societal_keywords)
    has_economic = any(k in text for k in economic_keywords)
    has_physical = any(k in text for k in physical_keywords)
    
    if has_societal:
        return 'Societal & Reputational'
    elif has_economic:
        return 'Economic & Allocative'
    elif has_physical:
        return 'Physical Safety'
    else:
        return 'Other/Unspecified'

aiid['Harm_Group'] = aiid.apply(map_harm_category, axis=1)

# Filter for analysis
analysis_df = aiid[aiid['Harm_Group'] != 'Other/Unspecified'].copy()

print("\nHarm Group Distribution (Known):")
print(analysis_df['Harm_Group'].value_counts())

# 5. Statistical Analysis & Plotting
if not analysis_df.empty:
    contingency = pd.crosstab(analysis_df['Harm_Group'], analysis_df['AI_Type'])
    contingency_pct = pd.crosstab(analysis_df['Harm_Group'], analysis_df['AI_Type'], normalize='columns') * 100
    
    print("\nContingency Table (Counts):")
    print(contingency)
    print("\nContingency Table (Column %):")
    print(contingency_pct.round(2))
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}\nP-value: {p:.4e}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plot_data = contingency_pct.reset_index().melt(id_vars='Harm_Group', var_name='AI_Type', value_name='Percentage')
    sns.barplot(data=plot_data, x='Harm_Group', y='Percentage', hue='AI_Type')
    plt.title('Harm Category Profile: Generative vs Discriminative AI')
    plt.ylabel('Percentage of Incidents (%)')
    plt.xlabel('Harm Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Text mining yielded no categorized harms.")
