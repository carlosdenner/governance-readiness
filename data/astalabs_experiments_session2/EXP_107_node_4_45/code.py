import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_experiment():
    # Robust file loading
    filename = 'astalabs_discovery_all_data.csv'
    possible_paths = [filename, f'../{filename}']
    file_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if file_path is None:
        print(f"Error: Could not find {filename} in current or parent directory.")
        return

    print(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Filter for EO 13960 Scored data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 Scored rows: {len(df_eo)}")
    
    # Combine text columns for keyword search (handling NaNs)
    df_eo['text_content'] = df_eo['2_use_case_name'].fillna('').astype(str) + ' ' + df_eo['11_purpose_benefits'].fillna('').astype(str)
    df_eo['text_content'] = df_eo['text_content'].str.lower()
    
    # Define Keywords
    genai_keywords = ['generative', 'llm', 'language model', 'chatbot', 'summariz', 'text generation']
    predictive_keywords = ['predict', 'detect', 'classify', 'vision', 'risk model']
    
    # helper function
    def has_keyword(text, keywords):
        return any(k in text for k in keywords)
    
    # Create flags
    df_eo['is_genai'] = df_eo['text_content'].apply(lambda x: has_keyword(x, genai_keywords))
    
    # Control group: Predictive keywords AND NOT GenAI (to isolate traditional AI)
    df_eo['is_predictive'] = df_eo['text_content'].apply(lambda x: has_keyword(x, predictive_keywords)) & (~df_eo['is_genai'])
    
    genai_group = df_eo[df_eo['is_genai']]
    pred_group = df_eo[df_eo['is_predictive']]
    
    print(f"\nSample Sizes:\n  GenAI: {len(genai_group)}\n  Traditional Predictive: {len(pred_group)}")
    
    if len(genai_group) < 5 or len(pred_group) < 5:
        print("Warning: Small sample sizes may affect statistical validity.")

    # Compliance Analysis Targets
    # 59_ai_notice: Did the agency provide notice?
    # 67_opt_out: Did the agency provide opt-out/consent mechanisms?
    targets = {
        'Notice (Q59)': '59_ai_notice',
        'Consent (Q67)': '67_opt_out'
    }
    
    results = []
    
    for label, col in targets.items():
        print(f"\n--- Analyzing {label} ---")
        
        # Check for unique values to ensure mapping is correct
        # print(f"Unique values in {col}: {df_eo[col].unique()}")
        
        def is_compliant(val):
            if pd.isna(val):
                return 0
            return 1 if str(val).strip().lower() == 'yes' else 0

        # GenAI stats
        genai_compliant = genai_group[col].apply(is_compliant).sum()
        genai_total = len(genai_group)
        genai_rate = (genai_compliant / genai_total) if genai_total > 0 else 0
        
        # Predictive stats
        pred_compliant = pred_group[col].apply(is_compliant).sum()
        pred_total = len(pred_group)
        pred_rate = (pred_compliant / pred_total) if pred_total > 0 else 0
        
        print(f"  GenAI: {genai_compliant}/{genai_total} ({genai_rate*100:.1f}%)")
        print(f"  Predictive: {pred_compliant}/{pred_total} ({pred_rate*100:.1f}%)")
        
        # Fisher's Exact Test
        # Contingency Table: [[GenAI_Yes, GenAI_No], [Pred_Yes, Pred_No]]
        table = [
            [genai_compliant, genai_total - genai_compliant],
            [pred_compliant, pred_total - pred_compliant]
        ]
        
        odds_ratio, p_val = stats.fisher_exact(table)
        print(f"  Fisher's Exact Test p-value: {p_val:.4f}")
        
        results.append({
            'Metric': label,
            'GenAI_Rate': genai_rate * 100,
            'Pred_Rate': pred_rate * 100,
            'P_Value': p_val
        })

    # Visualization
    labels = [r['Metric'] for r in results]
    genai_vals = [r['GenAI_Rate'] for r in results]
    pred_vals = [r['Pred_Rate'] for r in results]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, genai_vals, width, label='GenAI (Emerging)', color='#d62728')
    rects2 = ax.bar(x + width/2, pred_vals, width, label='Predictive (Traditional)', color='#1f77b4')
    
    ax.set_ylabel('Compliance Rate (%)')
    ax.set_title('Governance Lag: Compliance Rates for GenAI vs Traditional AI')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    
    # Add p-values on chart
    for i, r in enumerate(results):
        p_text = f"p={r['P_Value']:.3f}"
        # Position text above the higher bar
        h = max(r['GenAI_Rate'], r['Pred_Rate'])
        ax.text(i, h + 2, p_text, ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_experiment()