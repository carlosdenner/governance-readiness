import json
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
import matplotlib.pyplot as plt
import os

# Define file path (one level up as per instructions)
file_path = '../step3_enrichments.json'

# Check if file exists, otherwise try current directory (fallback)
if not os.path.exists(file_path):
    file_path = 'step3_enrichments.json'

print(f"Loading dataset from: {file_path}")

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Successfully loaded {len(data)} records.")
    
    # Process data into a list of dictionaries for DataFrame creation
    processed_rows = []
    
    for entry in data:
        # Extract failure mode
        f_mode = entry.get('failure_mode', 'unknown')
        
        # Extract sub-competencies
        # Metadata mentions 'sub_competency_ids' or 'llm_sub_competencies'. 
        # We check which one is a list or parseable string.
        sc_ids = entry.get('sub_competency_ids', [])
        if not sc_ids:
            sc_ids = entry.get('llm_sub_competencies', [])
            
        # Ensure sc_ids is a list
        if isinstance(sc_ids, str):
            # formatting might be "['TR-1', 'IR-2']" or "TR-1;IR-2"
            if sc_ids.startswith('['):
                try:
                    sc_ids = eval(sc_ids)
                except:
                    sc_ids = []
            elif ';' in sc_ids:
                sc_ids = sc_ids.split(';')
            else:
                sc_ids = [sc_ids]
        
        if not isinstance(sc_ids, list):
            sc_ids = []
            
        # Calculate Integration Ratio
        # Heuristic: TR-xx is Trust, IR-xx is Integration
        ir_count = sum(1 for x in sc_ids if 'IR-' in str(x))
        tr_count = sum(1 for x in sc_ids if 'TR-' in str(x))
        total_count = ir_count + tr_count
        
        if total_count > 0:
            integration_ratio = ir_count / total_count
        else:
            integration_ratio = np.nan # No competencies mapped
            
        processed_rows.append({
            'case_study_id': entry.get('case_study_id'),
            'failure_mode': f_mode,
            'integration_ratio': integration_ratio,
            'total_competencies': total_count
        })
    
    df = pd.DataFrame(processed_rows)
    
    # Drop rows with no competencies mapped if any (though we expect them to have mappings)
    df_clean = df.dropna(subset=['integration_ratio'])
    
    # Categorize Failure Modes
    # Group 1: Prevention
    # Group 2: Post-Prevention (Detection, Response)
    
    group_prevention = df_clean[df_clean['failure_mode'] == 'prevention_failure']
    group_post_prevention = df_clean[df_clean['failure_mode'].isin(['detection_failure', 'response_failure'])]
    
    # Summary Stats
    print("\n--- Descriptive Statistics ---")
    print(f"Total Incidents with Mapped Competencies: {len(df_clean)}")
    
    print(f"\nGroup 1: Prevention Failure (n={len(group_prevention)})")
    if not group_prevention.empty:
        print(f"Mean Integration Ratio: {group_prevention['integration_ratio'].mean():.4f}")
        print(f"Std Dev: {group_prevention['integration_ratio'].std():.4f}")
    
    print(f"\nGroup 2: Post-Prevention (Detection/Response) (n={len(group_post_prevention)})")
    if not group_post_prevention.empty:
        print(f"Mean Integration Ratio: {group_post_prevention['integration_ratio'].mean():.4f}")
        print(f"Std Dev: {group_post_prevention['integration_ratio'].std():.4f}")
        
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.boxplot([group_prevention['integration_ratio'], group_post_prevention['integration_ratio']], 
                labels=['Prevention', 'Post-Prevention (Detect/Respond)'])
    plt.title('Integration Competency Ratio by Failure Mode')
    plt.ylabel('Integration Ratio (0=All Trust, 1=All Integration)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Statistical Test
    print("\n--- Statistical Test Results ---")
    if len(group_prevention) > 1 and len(group_post_prevention) > 1:
        u_stat, p_val = mannwhitneyu(group_prevention['integration_ratio'], 
                                     group_post_prevention['integration_ratio'], 
                                     alternative='two-sided')
        print(f"Mann-Whitney U Statistic: {u_stat}")
        print(f"P-Value: {p_val:.5f}")
        
        if p_val < 0.05:
            print("Result: Statistically Significant Difference.")
        else:
            print("Result: No Statistically Significant Difference.")
    else:
        print("Insufficient sample size in one or both groups to perform Mann-Whitney U test.")
        print("Note: Previous exploration indicated a severe skew (51/52 prevention failures).")

except Exception as e:
    print(f"An error occurred: {e}")
