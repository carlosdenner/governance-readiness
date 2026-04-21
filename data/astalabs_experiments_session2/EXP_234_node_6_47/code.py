import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import numpy as np
import re
import os

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
filepath = filename if os.path.exists(filename) else os.path.join('..', filename)

print(f"Loading dataset from {filepath}...")
try:
    df = pd.read_csv(filepath, low_memory=False)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)} rows")

# Identify columns
cols = aiid_df.columns
autonomy_col = next((c for c in cols if 'Autonomy Level' in c), None)
tech_col = next((c for c in cols if 'Known AI Technology' in c), None)

print(f"Using columns: Autonomy='{autonomy_col}', Tech='{tech_col}'")

# --- 1. Clean Autonomy Level ---
# Observed values: "Autonomy1", "Autonomy3", etc.
def map_autonomy(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    
    # Regex to capture "Autonomy" followed by a digit
    match = re.search(r'Autonomy(\d+)', val_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Fallback: simple digit check if regex fails
    if val_str.isdigit():
        return int(val_str)
        
    return None

if autonomy_col:
    aiid_df['Autonomy_Score'] = aiid_df[autonomy_col].apply(map_autonomy)
else:
    aiid_df['Autonomy_Score'] = np.nan

# --- 2. Clean Technology ---
# Observed values: "Transformer", "Face Detection", "Visual Object Detection, Image Segmentation"
def map_technology(val):
    if pd.isna(val):
        return 'Other'
    val_str = str(val).lower()
    
    # Vision Keywords
    vision_keys = ['vision', 'image', 'face', 'facial', 'camera', 'video', 'detection', 'segmentation', 'recognition', 'ocr', 'optical character']
    # Language Keywords
    lang_keys = ['language', 'text', 'nlp', 'translation', 'transformer', 'bert', 'gpt', 'llm', 'chat', 'dialogue', 'document', 'summary', 'speech', 'voice']
    # Robotics Keywords (Physical agents)
    robot_keys = ['robot', 'drone', 'vehicle', 'car', 'autonomous driving', 'self-driving', 'uav', 'physical', 'manipulation', 'navigation', 'tesla', 'waymo', 'cruise']

    # Check Robotics first (often involves vision, but distinct by physical nature)
    if any(k in val_str for k in robot_keys):
        return 'Robotics'
    # Check Vision
    elif any(k in val_str for k in vision_keys):
        return 'Computer Vision'
    # Check Language
    elif any(k in val_str for k in lang_keys):
        return 'Language'
    else:
        return 'Other'

if tech_col:
    aiid_df['Tech_Category'] = aiid_df[tech_col].apply(map_technology)
else:
    aiid_df['Tech_Category'] = 'Other'

# --- 3. Analysis Filter ---
analysis_df = aiid_df.dropna(subset=['Autonomy_Score']).copy()
analysis_df = analysis_df[analysis_df['Tech_Category'].isin(['Robotics', 'Computer Vision', 'Language'])]

print(f"\nData points available for analysis: {len(analysis_df)}")
counts = analysis_df['Tech_Category'].value_counts()
print(counts)

# --- 4. Statistical Test & Visualization ---
groups = [analysis_df[analysis_df['Tech_Category'] == t]['Autonomy_Score'].values for t in ['Robotics', 'Computer Vision', 'Language']]

# Only proceed if we have data for comparison
if len(analysis_df) > 0 and len(groups) == 3:
    # Perform Kruskal-Wallis only if each group has at least one sample, otherwise print warning
    valid_groups = [g for g in groups if len(g) > 0]
    if len(valid_groups) > 1:
        stat, p = kruskal(*valid_groups)
        print(f"\nKruskal-Wallis H-test results:")
        print(f"Statistic: {stat:.4f}")
        print(f"p-value: {p:.4e}")
        if p < 0.05:
            print("Result: Significant difference detected.")
        else:
            print("Result: No significant difference.")
    else:
        print("\nNot enough groups for statistical comparison.")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Tech_Category', y='Autonomy_Score', data=analysis_df, 
                order=['Robotics', 'Computer Vision', 'Language'], palette='viridis')
    plt.title('Distribution of Autonomy Levels by AI Technology')
    plt.ylabel('Autonomy Score')
    plt.xlabel('Technology Domain')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("\nInsufficient data to generate plot or stats.")
