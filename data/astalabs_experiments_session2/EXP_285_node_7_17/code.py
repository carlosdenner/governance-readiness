import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re
import os

# Handle potential file path issues
filename = 'astalabs_discovery_all_data.csv'
if not os.path.exists(filename):
    if os.path.exists(f'../{filename}'):
        filename = f'../{filename}'
    else:
        # Fallback to searching, though unexpected based on prompt
        pass

print(f"Loading dataset from {filename}...")
df = pd.read_csv(filename, low_memory=False)

# Identify where the ATLAS case data resides
# We need 'name', 'summary', and 'techniques'

# Check 'atlas_cases' source_table
atlas_rows = df[df['source_table'] == 'atlas_cases'].copy()
print(f"Rows in atlas_cases: {len(atlas_rows)}")

# Check if techniques are populated in atlas_cases
techniques_populated_atlas = atlas_rows['techniques'].notna().sum()
print(f"Non-null techniques in atlas_cases: {techniques_populated_atlas}")

working_df = pd.DataFrame()

if techniques_populated_atlas > 10:
    working_df = atlas_rows
else:
    # Check 'step3_incident_coding'
    step3_rows = df[df['source_table'] == 'step3_incident_coding'].copy()
    techniques_populated_step3 = step3_rows['techniques'].notna().sum()
    print(f"Rows in step3_incident_coding: {len(step3_rows)}")
    print(f"Non-null techniques in step3_incident_coding: {techniques_populated_step3}")
    
    if techniques_populated_step3 > 10:
        working_df = step3_rows
    else:
        # Fallback: search entire dataframe for rows with 'techniques' and 'name' that look like ATLAS cases
        # ATLAS cases usually have a 'case_id' or 'summary'
        print("Searching entire dataframe for valid ATLAS entries...")
        mask = df['techniques'].notna() & (df['name'].notna() | df['summary'].notna())
        working_df = df[mask].copy()
        # Filter out rows that might be strictly incidents if they don't look like ATLAS cases, 
        # but for this specific dataset, techniques are primarily for ATLAS.
        print(f"Found {len(working_df)} rows with techniques populated.")

if len(working_df) == 0:
    print("No data found with populated techniques.")
else:
    # Function to categorize case type
    def categorize_system(row):
        # Combine name and summary for keyword search
        text_parts = []
        if pd.notna(row.get('name')):
            text_parts.append(str(row['name']))
        if pd.notna(row.get('summary')):
            text_parts.append(str(row['summary']))
        
        text = " ".join(text_parts).lower()
        
        gen_keywords = ['llm', 'gpt', 'genai', 'diffusion', 'chatbot', 'generative', 'language model', 'text generation', 'chatgpt', 'openai', 'bard', 'bing chat']
        cv_keywords = ['image', 'face', 'facial', 'recognition', 'vehicle', 'camera', 'vision', 'object detection', 'video', 'surveillance', 'yolo', 'pixel', 'tesla', 'driving']
        
        is_gen = any(k in text for k in gen_keywords)
        is_cv = any(k in text for k in cv_keywords)
        
        if is_gen and not is_cv:
            return 'Generative AI'
        elif is_cv and not is_gen:
            return 'Computer Vision'
        elif is_gen and is_cv:
            # Heuristic: if it mentions both, check which is the primary subject. 
            # For simplicity in this experiment, we might classify as Mixed or check count.
            # Let's try to see if "generative" appears more often or specific strong keywords.
            return 'Mixed/Ambiguous'
        else:
            return 'Other'

    # Function to count techniques
    def count_techniques(val):
        if pd.isna(val) or str(val).strip() == '':
            return 0
        val_str = str(val)
        # Remove brackets if present (e.g. "['T1', 'T2']")
        if val_str.strip().startswith('[') and val_str.strip().endswith(']'):
             # simple strip
             val_str = val_str.strip()[1:-1]
        
        # Split by comma or semicolon
        tokens = re.split(r'[,;]\s*', val_str)
        # Filter out empty strings and quotes
        clean_tokens = [t.strip().strip("'").strip('"') for t in tokens if t.strip()]
        return len(clean_tokens)

    working_df['system_category'] = working_df.apply(categorize_system, axis=1)
    working_df['technique_count'] = working_df['techniques'].apply(count_techniques)

    # Analysis Groups
    gen_ai = working_df[working_df['system_category'] == 'Generative AI']
    comp_vis = working_df[working_df['system_category'] == 'Computer Vision']

    print(f"\nCounts:\nGenerative AI: {len(gen_ai)}\nComputer Vision: {len(comp_vis)}\nOther/Mixed: {len(working_df) - len(gen_ai) - len(comp_vis)}")

    if len(gen_ai) > 1 and len(comp_vis) > 1:
        # Descriptive Stats
        print("\nGenerative AI - Technique Count Stats:")
        print(gen_ai['technique_count'].describe())
        print("\nComputer Vision - Technique Count Stats:")
        print(comp_vis['technique_count'].describe())

        # T-Test
        t_stat, p_val = stats.ttest_ind(gen_ai['technique_count'], comp_vis['technique_count'], equal_var=False)
        print(f"\nT-test Result: Statistic={t_stat:.4f}, p-value={p_val:.4f}")
        
        if p_val < 0.05:
            print("Conclusion: Significant difference in adversarial complexity.")
        else:
            print("Conclusion: No significant difference observed.")

        # Plot
        plt.figure(figsize=(8, 6))
        plt.boxplot([gen_ai['technique_count'], comp_vis['technique_count']], labels=['Generative AI', 'Computer Vision'])
        plt.title('Adversarial Complexity: Techniques per Case')
        plt.ylabel('Count of Techniques')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("Insufficient data in one or both categories to perform statistical testing.")