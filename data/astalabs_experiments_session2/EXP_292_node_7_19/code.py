import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

print(f"Total AIID incidents: {len(aiid_df)}")

# 1. Define Sector Filtering
# We need to identify Healthcare vs Financial sectors
# Let's clean the column first
aiid_df['Sector of Deployment'] = aiid_df['Sector of Deployment'].fillna('').astype(str)

def classify_sector(val):
    val_lower = val.lower()
    if 'health' in val_lower or 'medicine' in val_lower or 'medical' in val_lower:
        return 'Healthcare'
    elif 'financ' in val_lower or 'bank' in val_lower or 'trading' in val_lower:
        return 'Financial'
    else:
        return 'Other'

aiid_df['target_sector'] = aiid_df['Sector of Deployment'].apply(classify_sector)

# Filter only for the target sectors
analysis_df = aiid_df[aiid_df['target_sector'].isin(['Healthcare', 'Financial'])].copy()
print(f"Incidents in target sectors (Healthcare/Financial): {len(analysis_df)}")
print(analysis_df['target_sector'].value_counts())

# 2. Define Technology Classification
# Classify 'Known AI Technology' into 'Vision', 'Language', or 'Other'
analysis_df['Known AI Technology'] = analysis_df['Known AI Technology'].fillna('').astype(str)

def classify_tech(val):
    val_lower = val.lower()
    
    # Vision keywords
    vision_keywords = ['vision', 'image', 'video', 'facial', 'face', 'recognition', 'detection', 'surveillance', 'camera']
    is_vision = any(k in val_lower for k in vision_keywords)
    
    # Language keywords
    language_keywords = ['language', 'text', 'nlp', 'chat', 'llm', 'generative', 'translation', 'speech', 'voice', 'chatbot']
    is_language = any(k in val_lower for k in language_keywords)
    
    if is_vision and not is_language:
        return 'Computer Vision'
    elif is_language and not is_vision:
        return 'Language/Text'
    elif is_vision and is_language:
        return 'Multimodal/Both' # Or prioritize one based on hypothesis, but let's keep separate for clarity
    else:
        return 'Unspecified/Other'

analysis_df['tech_modality'] = analysis_df['Known AI Technology'].apply(classify_tech)

# Filter out Unspecified/Other/Multimodal to test the specific hypothesis strictly between Vision and Language
# Hypothesis: Healthcare -> Vision, Financial -> Language
strict_df = analysis_df[analysis_df['tech_modality'].isin(['Computer Vision', 'Language/Text'])].copy()

print(f"Incidents with clear Vision/Language classification: {len(strict_df)}")
print(strict_df.groupby(['target_sector', 'tech_modality']).size())

# 3. Create Contingency Table
contingency_table = pd.crosstab(strict_df['target_sector'], strict_df['tech_modality'])
print("\nContingency Table:")
print(contingency_table)

# 4. Chi-Square Test
if not contingency_table.empty and contingency_table.shape == (2, 2):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")
    
    # 5. Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
    plt.title('AI Incident Counts: Sector vs. Technology Modality')
    plt.ylabel('Sector')
    plt.xlabel('Technology Modality')
    plt.show()
    
    # Calculate percentages for interpretation
    row_sums = contingency_table.sum(axis=1)
    props = contingency_table.div(row_sums, axis=0)
    print("\nProportions (Row-wise):")
    print(props)
else:
    print("Insufficient data for Chi-Square test (need 2x2 table).")
