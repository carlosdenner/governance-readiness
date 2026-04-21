import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# 1. Load Data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# 2. Classification Logic

# Helper function for keyword matching
def classify_modality(text):
    if pd.isna(text):
        return 'Unknown'
    text = text.lower()
    
    # Keywords
    vision_keys = ['vision', 'image', 'face', 'facial', 'video', 'surveillance', 'object detection', 'cnn', 'convolutional']
    lang_keys = ['language', 'text', 'speech', 'translation', 'conversation', 'chatbot', 'transformer', 'nlp', 'generative', 'llm']
    
    is_vision = any(k in text for k in vision_keys)
    is_lang = any(k in text for k in lang_keys)
    
    if is_vision and not is_lang:
        return 'Vision'
    elif is_lang and not is_vision:
        return 'Language'
    elif is_vision and is_lang:
        return 'Multimodal'
    else:
        return 'Other'

def classify_failure(text):
    if pd.isna(text):
        return 'Unknown'
    text = text.lower()
    
    # Keywords based on prompt and common taxonomy
    # Robustness: adversarial, perturbation, sensitivity
    robust_keys = ['adversarial', 'perturbation', 'robustness', 'sensitivity', 'evasion', 'poisoning']
    
    # Output/Content: hallucination, toxic, offensive, inappropriate
    # Adding 'bias' here as it often relates to content output in social contexts, though distinct.
    # The prompt specifically mentioned 'hallucination, toxic output'.
    # Looking at unique values from debug: 'Inappropriate Training Content', 'Problematic Input' (maybe?)
    # Let's stick to the prompt's examples and close synonyms.
    content_keys = ['hallucination', 'toxic', 'offensive', 'inappropriate', 'content', 'hate', 'slur', 'misinformation', 'unsafe', 'bias']
    
    is_robust = any(k in text for k in robust_keys)
    is_content = any(k in text for k in content_keys)
    
    if is_robust and not is_content:
        return 'Robustness'
    elif is_content and not is_robust:
        return 'Output/Content'
    elif is_robust and is_content:
        return 'Mixed'
    else:
        return 'Other'

# Apply classification
aiid['Modality'] = aiid['Known AI Technology'].apply(classify_modality)
aiid['Failure_Type'] = aiid['Known AI Technical Failure'].apply(classify_failure)

# Filter for relevant groups
subset = aiid[
    (aiid['Modality'].isin(['Vision', 'Language'])) &
    (aiid['Failure_Type'].isin(['Robustness', 'Output/Content']))
].copy()

print("Subset Shape:", subset.shape)
print(subset.groupby(['Modality', 'Failure_Type']).size())

# 3. Statistical Test
if len(subset) > 0:
    contingency = pd.crosstab(subset['Modality'], subset['Failure_Type'])
    print("\n--- Contingency Table ---")
    print(contingency)
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Calculate Row Percentages for clarity
    row_props = pd.crosstab(subset['Modality'], subset['Failure_Type'], normalize='index')
    print("\nRow Proportions (Frequency of Failure Type given Modality):")
    print(row_props)

    # 4. Visualization
    # Using a Stacked Bar Chart as a simpler alternative to Sankey for categorical distribution
    plt.figure(figsize=(10, 6))
    row_props.plot(kind='bar', stacked=True, color=['orange', 'skyblue'])
    plt.title('Distribution of Technical Failures by AI Modality')
    plt.xlabel('AI Modality')
    plt.ylabel('Proportion of Incidents')
    plt.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data matched the criteria to perform analysis.")
