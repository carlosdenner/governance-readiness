import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# --- Load Dataset Robustly ---
filename = 'astalabs_discovery_all_data.csv'
if not os.path.exists(filename):
    if os.path.exists('../' + filename):
        filename = '../' + filename

print(f"Loading dataset from: {filename}")
try:
    df = pd.read_csv(filename, low_memory=False)
except FileNotFoundError:
    print(f"Error: {filename} not found.")
    raise

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid)} rows")

# --- Column Identification ---
def find_col(df, keyword):
    matches = [c for c in df.columns if keyword.lower() in str(c).lower()]
    return matches[0] if matches else None

col_tech = find_col(aiid, 'Known AI Technology')
col_harm_type = find_col(aiid, 'harm_type') # distinct from 'Harm Domain'
col_primary_harm = find_col(aiid, 'primary_harm_types')
col_reports = find_col(aiid, 'reports')
col_desc = find_col(aiid, 'description')
col_summary = find_col(aiid, 'summary')

print("Identified Columns:")
print(f"  Tech: {col_tech}")
print(f"  Harm Type: {col_harm_type}")
print(f"  Primary Harm: {col_primary_harm}")
print(f"  Reports/Desc: {col_reports} / {col_desc} / {col_summary}")

# --- 1. Tech Classification ---
def classify_tech(text):
    if pd.isna(text):
        return 'Unknown'
    text = str(text).lower()
    
    gen_keywords = [
        'generative', 'llm', 'gpt', 'chatbot', 'chat', 'transformer', 'diffusion', 
        'language model', 'text-to', 'dall-e', 'midjourney', 'stable diffusion', 
        'deepfake', 'synthetic', 'stylegan', 'voice clone'
    ]
    
    disc_keywords = [
        'classifier', 'classification', 'regression', 'decision tree', 'recognition', 
        'detection', 'recommendation', 'predictive', 'scoring', 'ranking', 
        'surveillance', 'computer vision', 'object detection', 'facial', 
        'algorithm', 'screening', 'monitoring', 'neural network', 'deep learning'
    ]
    
    # Prioritize Generative
    if any(k in text for k in gen_keywords):
        return 'Generative'
    if any(k in text for k in disc_keywords):
        return 'Discriminative'
    return 'Unclassified'

aiid['Tech_Class'] = aiid[col_tech].apply(classify_tech)

# --- 2. Harm Classification ---
# Strategy: Try structured columns first, fall back to text search
def classify_harm(row):
    # Gather all available text
    structured_text = ""
    if col_harm_type and pd.notna(row[col_harm_type]):
        structured_text += str(row[col_harm_type]) + " "
    if col_primary_harm and pd.notna(row[col_primary_harm]):
        structured_text += str(row[col_primary_harm]) + " "
    
    unstructured_text = ""
    if col_reports and pd.notna(row[col_reports]):
        unstructured_text += str(row[col_reports]) + " "
    if col_desc and pd.notna(row[col_desc]):
        unstructured_text += str(row[col_desc]) + " "
    if col_summary and pd.notna(row[col_summary]):
        unstructured_text += str(row[col_summary]) + " "
    
    # Prefer structured if meaningful, else use unstructured
    text = (structured_text + " " + unstructured_text).lower()
    
    # Keywords
    grp_a_keywords = [ # Reputational / Psychological
        'reputation', 'psychological', 'emotional', 'defamation', 'libel', 'slander', 
        'harassment', 'dignity', 'stress', 'mental', 'stigma', 'humiliation', 
        'offensive', 'hate speech', 'bias', 'discrimination', 'racist', 'sexist', 'stereotype'
    ]
    
    grp_b_keywords = [ # Economic / Physical / Opportunity
        'economic', 'financial', 'monetary', 'money', 'property', 'asset', 'employment', 
        'job', 'hiring', 'opportunity', 'access', 'physical', 'bodily', 'injury', 'death', 
        'kill', 'safety', 'health', 'medical', 'credit', 'insurance', 'housing', 'arrest'
    ]
    
    has_a = any(k in text for k in grp_a_keywords)
    has_b = any(k in text for k in grp_b_keywords)
    
    if has_a and not has_b:
        return 'Reputational/Psychological'
    elif has_b and not has_a:
        return 'Economic/Physical/Opportunity'
    elif has_a and has_b:
        return 'Mixed'
    else:
        return 'Other'

aiid['Harm_Category'] = aiid.apply(classify_harm, axis=1)

# --- 3. Analysis ---
analysis_subset = aiid[
    (aiid['Tech_Class'].isin(['Generative', 'Discriminative'])) & 
    (aiid['Harm_Category'].isin(['Reputational/Psychological', 'Economic/Physical/Opportunity']))
]

contingency_table = pd.crosstab(analysis_subset['Tech_Class'], analysis_subset['Harm_Category'])

print("\n--- Analysis Summary ---")
print(f"Total AIID Incidents: {len(aiid)}")
print(f"Categorized Incidents: {len(analysis_subset)}")
print(f"Generative Count: {len(analysis_subset[analysis_subset['Tech_Class'] == 'Generative'])}")
print(f"Discriminative Count: {len(analysis_subset[analysis_subset['Tech_Class'] == 'Discriminative'])}")

print("\n--- Contingency Table ---")
print(contingency_table)

# --- 4. Statistics ---
if not contingency_table.empty and contingency_table.shape == (2, 2):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    props = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    print("\n--- Proportions (%) ---")
    print(props)
    
    ax = props.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='RdBu')
    plt.title('Harm Category Distribution by AI Technology')
    plt.ylabel('Percentage')
    plt.xlabel('AI Technology')
    plt.legend(title='Harm Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient data for Chi-square test.")
