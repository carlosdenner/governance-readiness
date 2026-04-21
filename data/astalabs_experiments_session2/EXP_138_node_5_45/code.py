import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

# [debug] # Set up simple debug print
# def debug_print(msg):
#     print(f"[DEBUG] {msg}")

try:
    # Load the dataset
    file_path = '../astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

    # Filter for ATLAS cases
    atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
    
    print(f"Loaded ATLAS cases: {len(atlas_df)} records")

    # Define keywords for modality classification
    # Combining name and summary for search
    atlas_df['combined_text'] = (atlas_df['name'].fillna('') + ' ' + atlas_df['summary'].fillna('')).str.lower()
    
    vision_keywords = ['image', 'face', 'facial', 'recognition', 'camera', 'video', 'vision', 'pixel', 
                       'surveillance', 'biometric', 'object detection', 'yolo', 'glasses', 'patch', 
                       'traffic', 'sign', 'autonomous', 'driving', 'vehicle', 'tesla', 'lidar']
    
    language_keywords = ['text', 'language', 'translation', 'chat', 'bot', 'gpt', 'bert', 'llm', 
                         'dialogue', 'speech', 'email', 'phishing', 'spam', 'tweet', 'twitter', 
                         'sentiment', 'word', 'translate', 'completion']

    def classify_modality(text):
        is_vision = any(k in text for k in vision_keywords)
        is_language = any(k in text for k in language_keywords)
        
        if is_vision and not is_language:
            return 'Vision'
        elif is_language and not is_vision:
            return 'Language'
        elif is_vision and is_language:
            # Conflict resolution: check counts or default to Multimodal, but for this experiment let's try to discern
            # heuristic: if 'vision' or 'image' appears, it's likely vision even if it has text
            # For now mark as 'Mixed'
            return 'Mixed'
        else:
            return 'Other'

    atlas_df['modality'] = atlas_df['combined_text'].apply(classify_modality)
    
    # Classify tactics
    # Target: Evasion vs (Impact OR Exfiltration)
    atlas_df['tactics'] = atlas_df['tactics'].fillna('').str.lower()
    
    def check_tactic(tactic_str, target_list):
        return any(t in tactic_str for t in target_list)

    # "Evasion" usually appears as "Defense Evasion" in ATLAS or just "Evasion"
    atlas_df['has_evasion'] = atlas_df['tactics'].apply(lambda x: 'evasion' in x)
    # "Impact" or "Exfiltration"
    atlas_df['has_impact_exfil'] = atlas_df['tactics'].apply(lambda x: 'impact' in x or 'exfiltration' in x)

    # Filter for only Vision and Language for the hypothesis test
    analysis_df = atlas_df[atlas_df['modality'].isin(['Vision', 'Language'])].copy()
    
    # Generate Summary Stats
    modality_counts = analysis_df['modality'].value_counts()
    print("\n--- Modality Distribution ---")
    print(modality_counts)
    
    # Create Contingency Table for Evasion
    print("\n--- Hypothesis 1: Vision -> Evasion ---")
    ct_evasion = pd.crosstab(analysis_df['modality'], analysis_df['has_evasion'])
    print(ct_evasion)
    
    # Fisher's Exact Test for Evasion
    # We want to see if Vision has higher rate of True than Language
    # Contingency table structure roughly:
    #           False   True
    # Language  A       B
    # Vision    C       D
    # We compare proportions B/(A+B) vs D/(C+D)
    if 'Vision' in ct_evasion.index and 'Language' in ct_evasion.index:
        odds_ratio, p_value_evasion = stats.fisher_exact(ct_evasion)
        # Note: fisher_exact is for 2x2. 
        # Check if Vision is significantly different from Language
        vision_evasion_rate = analysis_df[analysis_df['modality']=='Vision']['has_evasion'].mean()
        language_evasion_rate = analysis_df[analysis_df['modality']=='Language']['has_evasion'].mean()
        print(f"Vision Evasion Rate: {vision_evasion_rate:.2%}")
        print(f"Language Evasion Rate: {language_evasion_rate:.2%}")
        print(f"Fisher's Exact Test p-value: {p_value_evasion:.4f}")
    else:
        print("Insufficient data for Evasion test.")

    # Create Contingency Table for Impact/Exfiltration
    print("\n--- Hypothesis 2: Language -> Impact/Exfiltration ---")
    ct_impact = pd.crosstab(analysis_df['modality'], analysis_df['has_impact_exfil'])
    print(ct_impact)
    
    if 'Vision' in ct_impact.index and 'Language' in ct_impact.index:
        odds_ratio_imp, p_value_impact = stats.fisher_exact(ct_impact)
        vision_impact_rate = analysis_df[analysis_df['modality']=='Vision']['has_impact_exfil'].mean()
        language_impact_rate = analysis_df[analysis_df['modality']=='Language']['has_impact_exfil'].mean()
        print(f"Vision Impact/Exfil Rate: {vision_impact_rate:.2%}")
        print(f"Language Impact/Exfil Rate: {language_impact_rate:.2%}")
        print(f"Fisher's Exact Test p-value: {p_value_impact:.4f}")
    else:
        print("Insufficient data for Impact/Exfil test.")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Evasion
    if not ct_evasion.empty:
        # Normalize to get percentages
        ct_evasion_norm = pd.crosstab(analysis_df['modality'], analysis_df['has_evasion'], normalize='index')
        if True in ct_evasion_norm.columns:
            ct_evasion_norm[True].plot(kind='bar', ax=axes[0], color=['orange', 'blue'], alpha=0.7)
        else:
             # Handle case where no True values exist
             ct_evasion_norm.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Rate of Evasion Tactics by Modality')
        axes[0].set_ylabel('Proportion of Cases')
        axes[0].set_ylim(0, 1.0)

    # Plot 2: Impact/Exfil
    if not ct_impact.empty:
        ct_impact_norm = pd.crosstab(analysis_df['modality'], analysis_df['has_impact_exfil'], normalize='index')
        if True in ct_impact_norm.columns:
            ct_impact_norm[True].plot(kind='bar', ax=axes[1], color=['orange', 'blue'], alpha=0.7)
        else:
             ct_impact_norm.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Rate of Impact/Exfiltration by Modality')
        axes[1].set_ylabel('Proportion of Cases')
        axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
