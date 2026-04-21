import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import sys
import os

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = '../astalabs_discovery_all_data.csv'

print(f"Loading dataset from {file_path}...")
try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents found: {len(aiid_df)}")

# Construct text_blob for analysis
# Check for potential text columns
potential_text_cols = ['title', 'description', 'summary', 'text']
text_cols = [c for c in potential_text_cols if c in aiid_df.columns]
print(f"Using text columns for mining: {text_cols}")

if not text_cols:
    print("No text columns found for mining. Available columns:", aiid_df.columns[:20])
    sys.exit(1)

# Fill NaNs with empty string and concatenate
aiid_df['text_blob'] = aiid_df[text_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()

# --- 1. Classify Incident Nature (Intentional vs Accidental) ---
intent_keywords = [
    'adversarial', 'attack', 'malicious', 'hacker', 'poisoning', 
    'evasion', 'deliberate', 'jailbreak', 'prompt injection', 'intentional'
]

def classify_intent(text):
    if any(k in text for k in intent_keywords):
        return 'Intentional'
    return 'Accidental'

aiid_df['Incident_Nature'] = aiid_df['text_blob'].apply(classify_intent)

# --- 2. Classify Harm Type (Tangible vs Intangible) ---
tangible_keywords = [
    'death', 'injury', 'kill', 'physical', 'safety', 'crash', 
    'collision', 'hurt', 'body', 'medical', 'accident', 'damage'
]
intangible_keywords = [
    'economic', 'reputation', 'financial', 'bias', 'discrimination', 
    'copyright', 'fraud', 'scam', 'money', 'loss', 'job', 'credit', 
    'defamation', 'privacy', 'surveillance', 'academic'
]

def classify_harm(text):
    has_tangible = any(k in text for k in tangible_keywords)
    has_intangible = any(k in text for k in intangible_keywords)
    
    if has_tangible and not has_intangible:
        return 'Tangible (Physical/Safety)'
    if has_intangible and not has_tangible:
        return 'Intangible (Economic/Reputational)'
    if has_tangible and has_intangible:
        return 'Ambiguous/Mixed'
    return 'Ambiguous/Other'

aiid_df['Harm_Type'] = aiid_df['text_blob'].apply(classify_harm)

# Filter for analysis
analysis_df = aiid_df[aiid_df['Harm_Type'].isin(['Tangible (Physical/Safety)', 'Intangible (Economic/Reputational)'])]
print(f"\nClassified Incidents for Analysis: {len(analysis_df)} (out of {len(aiid_df)})\n")

# --- 3. Statistical Analysis ---
contingency = pd.crosstab(analysis_df['Incident_Nature'], analysis_df['Harm_Type'])

print("--- Contingency Table ---")
print(contingency)

if contingency.size == 0 or contingency.sum().sum() == 0:
    print("\nInsufficient data for Chi-square test.")
else:
    # Calculate Row Percentages
    row_pcts = contingency.div(contingency.sum(axis=1), axis=0).mul(100).round(1)
    print("\n--- Row Percentages (Nature -> Harm Distribution) ---")
    print(row_pcts)

    # Chi-square Test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.5f}")
    
    if p < 0.05:
        print("\nResult: Statistically Significant Association.")
    else:
        print("\nResult: No Statistically Significant Association.")
