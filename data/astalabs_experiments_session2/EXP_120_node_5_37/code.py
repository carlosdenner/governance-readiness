import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# 1. Classify AI Technology
def classify_tech(tech_str):
    if pd.isna(tech_str):
        return 'Unknown'
    tech_str = str(tech_str).lower()
    
    gen_keywords = ['generative', 'gan', 'gpt', 'llm', 'diffusion', 'transformer', 'chatbot', 'language model', 'text generation', 'image generation', 'content generation', 'synthes']
    pred_keywords = ['regression', 'classification', 'classifier', 'recognition', 'detection', 'predictive', 'recommendation', 'filtering', 'scoring', 'assessment', 'computer vision', 'clustering']
    
    # Prioritize Generative because some predictive systems use transformers now, but in AIID context usually implies GenAI if explicit
    if any(k in tech_str for k in gen_keywords):
        return 'Generative'
    elif any(k in tech_str for k in pred_keywords):
        return 'Predictive'
    else:
        return 'Other'

aiid_df['Tech_Type'] = aiid_df['Known AI Technology'].apply(classify_tech)

# 2. Classify Harm Domain from Description (since structured labels are missing/boolean)
def classify_harm(text):
    if pd.isna(text):
        return 'Unknown'
    text = str(text).lower()
    
    # Keywords for Psych/Reputational
    psych_keywords = [
        'reputation', 'defamation', 'slander', 'libel', 'bias', 'discrimination', 'racist', 'sexist', 
        'slur', 'offensive', 'toxic', 'harassment', 'bullying', 'psychological', 'mental', 'stress', 
        'anxiety', 'trauma', 'dignity', 'privacy', 'surveillance', 'shaming', 'stereotype', 'misinformation', 'hallucination'
    ]
    
    # Keywords for Economic/Allocative
    econ_keywords = [
        'economic', 'financial', 'monetary', 'money', 'job', 'employment', 'hiring', 'firing', 
        'credit', 'loan', 'insurance', 'benefits', 'welfare', 'housing', 'allocation', 'resource', 
        'opportunity', 'access', 'price', 'market', 'fraud', 'theft', 'loss of funds', 'payment'
    ]
    
    has_psych = any(k in text for k in psych_keywords)
    has_econ = any(k in text for k in econ_keywords)
    
    if has_psych and not has_econ:
        return 'Psychological/Reputational'
    elif has_econ and not has_psych:
        return 'Economic/Allocative'
    elif has_psych and has_econ:
        return 'Mixed'
    else:
        return 'Other'

# Apply harm classification on 'description' column
aiid_df['Harm_Class'] = aiid_df['description'].apply(classify_harm)

# Filter out Unknown/Other/Mixed for cleaner analysis, or keep them if sample size is too small
analysis_df = aiid_df[
    (aiid_df['Tech_Type'].isin(['Generative', 'Predictive'])) & 
    (aiid_df['Harm_Class'].isin(['Psychological/Reputational', 'Economic/Allocative']))
]

# Create Contingency Table
contingency_table = pd.crosstab(analysis_df['Tech_Type'], analysis_df['Harm_Class'])

print("--- Contingency Table ---")
print(contingency_table)

# Perform Statistical Test
if contingency_table.size >= 4:
    # Fisher's exact test for 2x2, Chi2 for larger (though we filtered to 2x2)
    if contingency_table.shape == (2, 2):
        oddsratio, pvalue = stats.fisher_exact(contingency_table)
        test_name = "Fisher's Exact Test"
    else:
        chi2, pvalue, dof, expected = stats.chi2_contingency(contingency_table)
        test_name = "Chi-Square Test"
        
    print(f"\n{test_name} Results:")
    print(f"P-value: {pvalue:.5f}")
    if pvalue < 0.05:
        print("Result: Statistically Significant Association")
    else:
        print("Result: No Statistically Significant Association")
else:
    print("\nNot enough data for statistical test.")

# Plotting
if not contingency_table.empty:
    # Calculate percentages for better comparison
    contingency_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    
    ax = contingency_pct.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])
    plt.title('Harm Domain Distribution by AI Technology Type')
    plt.xlabel('AI Technology Type')
    plt.ylabel('Percentage of Incidents')
    plt.legend(title='Harm Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Add counts to bars
    for i, (idx, row) in enumerate(contingency_table.iterrows()):
        total = row.sum()
        if total > 0:
            y_pos = 0
            for col in contingency_table.columns:
                count = row[col]
                pct = (count / total) * 100
                if count > 0:
                    plt.text(i, y_pos + pct/2, f"{count}\n({pct:.1f}%)", ha='center', va='center', color='white')
                y_pos += pct
    
    plt.show()
