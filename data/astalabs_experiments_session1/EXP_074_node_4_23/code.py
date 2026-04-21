import json
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import os

# [debug] Print current directory and list files to locate dataset
# print(f"Current working directory: {os.getcwd()}")
# print(f"Files in current directory: {os.listdir('.')}")
# if os.path.exists('../'):
#     print(f"Files in parent directory: {os.listdir('../')}")

# Define file path (prioritizing parent directory as per instructions)
filename = 'step3_enrichments.json'
file_path = f"../{filename}"
if not os.path.exists(file_path):
    file_path = filename  # Fallback to current directory

if not os.path.exists(file_path):
    print(f"Error: {filename} not found in ../ or .")
else:
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Process data
    records = []
    for entry in data:
        # Extract relevant fields
        case_id = entry.get('case_study_id')
        harm = entry.get('harm_type')
        
        # Determine sub-competency count
        # Checking both 'sub_competency_ids' (primary) and handling potential formats
        ids = entry.get('sub_competency_ids', [])
        
        count = 0
        if isinstance(ids, list):
            count = len(ids)
        elif isinstance(ids, str):
            # If semicolon separated string
            if ids.strip():
                count = len([x for x in ids.split(';') if x.strip()])
        
        records.append({
            'case_study_id': case_id,
            'harm_type': harm,
            'sub_competency_count': count
        })
    
    df = pd.DataFrame(records)
    
    # Define groups
    df['group'] = df['harm_type'].apply(lambda x: 'Security' if x == 'security' else 'Non-Security')
    
    # Calculate Descriptive Statistics
    group_stats = df.groupby('group')['sub_competency_count'].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
    print("\n=== Descriptive Statistics by Group ===")
    print(group_stats)
    
    # Prepare samples for statistical test
    sec_counts = df[df['group'] == 'Security']['sub_competency_count']
    non_sec_counts = df[df['group'] == 'Non-Security']['sub_competency_count']
    
    # Perform Mann-Whitney U Test
    # Hypothesis: Security > Non-Security
    stat, p_val = mannwhitneyu(sec_counts, non_sec_counts, alternative='greater')
    
    print("\n=== Mann-Whitney U Test Results ===")
    print(f"Hypothesis: 'Security' incidents have more mapped sub-competencies than 'Non-Security'.")
    print(f"U-statistic: {stat}")
    print(f"P-value: {p_val:.5f}")
    
    alpha = 0.05
    if p_val < alpha:
        print("Result: Statistically Significant (Reject Null Hypothesis)")
    else:
        print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")

    # Visualization
    plt.figure(figsize=(10, 6))
    # Create boxplot data
    data_to_plot = [sec_counts, non_sec_counts]
    labels = [f'Security (n={len(sec_counts)})', f'Non-Security (n={len(non_sec_counts)})']
    
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red'))
    
    plt.title('Distribution of Missing Sub-Competencies per Incident')
    plt.ylabel('Count of Sub-Competencies')
    plt.xlabel('Incident Harm Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate means
    means = [sec_counts.mean(), non_sec_counts.mean()]
    for i, mean in enumerate(means, 1):
        plt.text(i, mean + 0.1, f'Mean: {mean:.2f}', 
                 horizontalalignment='center', color='darkblue', fontweight='bold')

    plt.show()