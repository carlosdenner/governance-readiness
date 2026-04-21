import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

print("Starting experiment: The Generative-Malice Link")

# Load dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID incidents: {len(aiid)}")

# Relevant columns
tech_col = 'Known AI Technology'
intent_col = 'Intentional Harm'

# Drop rows where technology or intent is missing
aiid_clean = aiid.dropna(subset=[tech_col, intent_col])
print(f"Incidents with known technology and intent: {len(aiid_clean)}")

# Define classification logic for Technology
def classify_tech(text):
    text = str(text).lower()
    # Generative keywords
    gen_keywords = [
        'generative', 'gan', 'language model', 'llm', 'gpt', 'diffusion', 'transformer', 
        'text-to-image', 'chatbot', 'deepfake', 'image generator', 'voice cloning', 
        'synthesizer', 'stylegan', 'midjourney', 'dall-e', 'stable diffusion', 'bert', 
        'chatgpt', 'creative', 'writing', 'art'
    ]
    # Discriminative keywords
    disc_keywords = [
        'classifier', 'classification', 'regression', 'decision tree', 'svm', 'support vector', 
        'recommendation', 'ranking', 'detection', 'recognition', 'predictive', 'scoring', 
        'computer vision', 'object detection', 'face recognition', 'neural network', 
        'deep learning', 'distributional learning', 'content-based filtering', 
        'collaborative filtering', 'segmentation', 'clustering', 'reinforcement learning', 
        'supervised', 'unsupervised', 'monitoring'
    ]
    
    # Check for generative first (hypothesis interest)
    if any(k in text for k in gen_keywords):
        return 'Generative'
    # Check for discriminative
    elif any(k in text for k in disc_keywords):
        return 'Discriminative'
    else:
        return 'Other'

# Apply classification
aiid_clean['Tech_Class'] = aiid_clean[tech_col].apply(classify_tech)

# Clean Intent column based on previous debug findings
def clean_intent(val):
    val_str = str(val).lower().strip()
    # Based on debug output: 'No. Not intentionally...', 'Yes. Intentionally...'
    if val_str.startswith('yes'):
        return 'Intentional'
    elif val_str.startswith('no'):
        return 'Unintentional'
    return 'Unknown'

aiid_clean['Intent_Class'] = aiid_clean[intent_col].apply(clean_intent)

# Filter for analysis
analysis_df = aiid_clean[
    (aiid_clean['Tech_Class'].isin(['Generative', 'Discriminative'])) & 
    (aiid_clean['Intent_Class'].isin(['Intentional', 'Unintentional']))
]

print(f"Rows used for analysis: {len(analysis_df)}")

# Generate contingency table
contingency = pd.crosstab(analysis_df['Tech_Class'], analysis_df['Intent_Class'])

print("\n--- Contingency Table ---")
print(contingency)

# Calculate proportions and run statistics
if not contingency.empty and contingency.shape == (2, 2):
    # Add Total column for rate calculation
    contingency['Total'] = contingency['Intentional'] + contingency['Unintentional']
    contingency['Intentional_Rate'] = contingency['Intentional'] / contingency['Total']
    
    print("\n--- Intentional Harm Rates ---")
    print(contingency[['Intentional', 'Total', 'Intentional_Rate']])

    # Statistical Test: Chi-square
    # Extract only the count data
    obs = contingency[['Intentional', 'Unintentional']].values
    chi2, p, dof, expected = chi2_contingency(obs)

    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Statistically significant difference found.")
        gen_rate = contingency.loc['Generative', 'Intentional_Rate']
        disc_rate = contingency.loc['Discriminative', 'Intentional_Rate']
        if gen_rate > disc_rate:
            print(f"Generative AI has a HIGHER rate of intentional harm ({gen_rate:.2%} vs {disc_rate:.2%}).")
        else:
            print(f"Generative AI has a LOWER rate of intentional harm ({gen_rate:.2%} vs {disc_rate:.2%}).")
    else:
        print("Result: No statistically significant difference found.")
else:
    print("Insufficient data for full 2x2 analysis (one or more categories might be empty).")
