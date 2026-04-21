import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = '../astalabs_discovery_all_data.csv'

try:
    df = pd.read_csv(file_path, low_memory=False)
except:
    # Fallback for very sparse CSVs if engine='c' fails (rare but possible)
    df = pd.read_csv(file_path, low_memory=False, engine='python')

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# Create a consolidated text field for classification
# We combine title, description, and Known AI Technology to maximize signal
text_cols = ['title', 'description', 'Known AI Technology']
for col in text_cols:
    if col not in aiid.columns:
        aiid[col] = ''
    else:
        aiid[col] = aiid[col].fillna('')

aiid['full_text'] = (aiid['title'] + ' ' + aiid['description'] + ' ' + aiid['Known AI Technology']).str.lower()

# --- Classification Functions ---

def classify_tech(text):
    # Robotics / Autonomous Vehicles
    if any(k in text for k in ['robot', 'autonomous vehicle', 'self-driving', 'drone', 'tesla', 'autopilot', 'waymo', 'cruise', 'uber', 'driverless']):
        return 'Robotics/AV'
    # Generative AI / LLM / Chatbots
    if any(k in text for k in ['generative', 'llm', 'gpt', 'chatgpt', 'diffusion', 'chatbot', 'midjourney', 'dall-e', 'language model', 'deepfake', 'text-to-image', 'stable diffusion', 'bert', 'transformer', 'hallucinat']):
        return 'Generative/LLM'
    return 'Other'

def classify_severity(text):
    # Level 3: Severe Physical Harm / Death
    if any(k in text for k in ['kill', 'death', 'fatal', 'died', 'suicide', 'loss of life', 'murder']):
        return 3
    # Level 2: Physical Injury / Property / Financial
    if any(k in text for k in ['injur', 'crash', 'collision', 'accident', 'damage', 'property', 'financial', 'money', 'theft', 'arrest', 'physical']):
        return 2
    # Level 1: Intangible / Social / Psychological
    if any(k in text for k in ['bias', 'discriminat', 'racis', 'sexis', 'privacy', 'surveillance', 'offensive', 'inappropriate', 'nudity', 'copyright', 'plagiaris', 'reputation', 'stereotyp', 'wrongful']):
        return 1
    # Level 0: Unclear / None specified (default base)
    return 0

# Apply Classifications
aiid['tech_category'] = aiid['full_text'].apply(classify_tech)
aiid['severity_score'] = aiid['full_text'].apply(classify_severity)

# Filter for comparison groups
comp_df = aiid[aiid['tech_category'].isin(['Robotics/AV', 'Generative/LLM'])].copy()

# Stats
robotics_scores = comp_df[comp_df['tech_category'] == 'Robotics/AV']['severity_score']
genai_scores = comp_df[comp_df['tech_category'] == 'Generative/LLM']['severity_score']

print(f"--- Classification Results ---")
print(f"Robotics/AV Samples: {len(robotics_scores)}")
print(f"Generative/LLM Samples: {len(genai_scores)}")
print(f"Total Classified: {len(comp_df)}")

print(f"\n--- Severity Statistics (Median) ---")
print(f"Robotics/AV: {robotics_scores.median()}")
print(f"Generative/LLM: {genai_scores.median()}")

# Mann-Whitney U Test
u_stat, p_val = stats.mannwhitneyu(robotics_scores, genai_scores, alternative='greater')
print(f"\n--- Hypothesis Test (Mann-Whitney U) ---")
print(f"Null: Severity(Robotics) <= Severity(GenAI)")
print(f"Alternative: Severity(Robotics) > Severity(GenAI)")
print(f"U-Statistic: {u_stat}")
print(f"P-Value: {p_val:.5e}")

# Visualization
# 1. Stacked Bar Chart (Proportions)
counts = comp_df.groupby(['tech_category', 'severity_score']).size().unstack(fill_value=0)
props = counts.div(counts.sum(axis=1), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Stacked Bar
props.plot(kind='bar', stacked=True, ax=axes[0], colormap='RdYlBu_r', alpha=0.85)
axes[0].set_title('Proportion of Severity Levels by Tech')
axes[0].set_ylabel('Proportion')
axes[0].set_xlabel('Technology')
axes[0].legend(title='Severity Score', labels=['0: None/Unclear', '1: Intangible', '2: Property/Injury', '3: Death/Severe'])

# 2. Boxplot (Distribution)
# We use a list of arrays for boxplot
data_to_plot = [robotics_scores, genai_scores]
axes[1].boxplot(data_to_plot, labels=['Robotics/AV', 'Generative/LLM'], patch_artist=True, 
                boxprops=dict(facecolor='lightblue'))
axes[1].set_title('Distribution of Severity Scores')
axes[1].set_ylabel('Severity Score (0-3)')
axes[1].yaxis.grid(True)

plt.tight_layout()
plt.show()

# Print counts for verification
print("\n--- Detailed Counts ---")
print(counts)
