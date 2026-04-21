import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys

# Ensure scipy is installed for statistical tests
try:
    import scipy.stats as stats
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "scipy"])
    import scipy.stats as stats

def run_experiment():
    # Attempt to locate the file based on the instruction
    file_path = '../step3_enrichments.json'
    if not os.path.exists(file_path):
        # Fallback to current directory if the note was context-dependent
        file_path = 'step3_enrichments.json'
    
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        return

    print(f"Loading dataset from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)

    records = []
    
    for entry in data:
        # Extract harm type
        harm_type = entry.get('harm_type', 'unknown')
        if not harm_type:
            harm_type = 'unknown'
        
        harm_norm = harm_type.strip().lower()
        is_security = (harm_norm == 'security')
        
        # Extract sub_competency_ids (handle list or string)
        sub_ids = entry.get('sub_competency_ids', [])
        if isinstance(sub_ids, str):
            # Clean and split if string (e.g. "IR-1; TR-2")
            sub_ids_list = [x.strip() for x in sub_ids.replace(';', ',').split(',') if x.strip()]
        elif isinstance(sub_ids, list):
            sub_ids_list = sub_ids
        else:
            sub_ids_list = []
            
        # Calculate counts for Integration (IR) and Trust (TR)
        ir_count = 0
        tr_count = 0
        
        for pid in sub_ids_list:
            pid_upper = pid.upper().strip()
            if pid_upper.startswith('IR'):
                ir_count += 1
            elif pid_upper.startswith('TR'):
                tr_count += 1
        
        total = ir_count + tr_count
        
        # Only include records that have mappable competencies
        if total > 0:
            ratio = ir_count / total
            records.append({
                'id': entry.get('case_study_id', 'unknown'),
                'harm_type': harm_norm,
                'category': 'Security' if is_security else 'Non-Security',
                'integration_ratio': ratio,
                'total_competencies': total
            })
            
    df = pd.DataFrame(records)
    
    # Summary Statistics
    print("=== Descriptive Statistics by Category ===")
    summary = df.groupby('category')['integration_ratio'].describe()
    print(summary)
    print("\n")
    
    # Prepare groups for statistical testing
    sec_data = df[df['category'] == 'Security']['integration_ratio']
    non_sec_data = df[df['category'] == 'Non-Security']['integration_ratio']
    
    # Mann-Whitney U Test (Non-parametric)
    u_stat, p_val_mw = stats.mannwhitneyu(sec_data, non_sec_data, alternative='two-sided')
    
    # Welch's T-test (Parametric, unequal variance)
    t_stat, p_val_ttest = stats.ttest_ind(sec_data, non_sec_data, equal_var=False)
    
    print("=== Statistical Test Results ===")
    print(f"Mann-Whitney U Statistic: {u_stat}")
    print(f"Mann-Whitney P-value: {p_val_mw:.5f}")
    print(f"Welch's T-Test Statistic: {t_stat:.4f}")
    print(f"Welch's T-Test P-value: {p_val_ttest:.5f}")
    
    if p_val_mw < 0.05:
        print("Result: Statistically significant difference found.")
    else:
        print("Result: No statistically significant difference found.")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    # Combine data for plotting
    plot_data = [sec_data, non_sec_data]
    labels = [f'Security (n={len(sec_data)})', f'Non-Security (n={len(non_sec_data)})']
    
    plt.boxplot(plot_data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor="lightblue"), 
                medianprops=dict(color="red", linewidth=1.5))
                
    plt.title('Integration Readiness Gap Ratio by Harm Category')
    plt.ylabel('Integration Ratio\n(IR Gaps / Total Gaps)')
    plt.xlabel('Harm Category')
    plt.ylim(-0.05, 1.05)  # Ratios are 0-1
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    run_experiment()