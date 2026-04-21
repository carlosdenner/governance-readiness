import pandas as pd
import scipy.stats as stats
import numpy as np

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(df_aiid)} rows")

# --- Step 1: define GenAI ---
# Keywords for Generative AI
genai_keywords = ['generative', 'llm', 'gpt', 'diffusion', 'language model', 'chatbot', 'deepfake', 'dall-e', 'midjourney']

def is_genai(text):
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    return any(keyword in text_lower for keyword in genai_keywords)

df_aiid['is_genai'] = df_aiid['Known AI Technology'].apply(is_genai)

# --- Step 2: Define Harm Type ---
# Based on previous debug, 'Tangible Harm' column contains specific strings.
# We define 'Tangible' as cases where harm definitively occurred.
# We define 'Intangible' as cases where it did not (near-misses, issues, or explicitly no tangible harm).

tangible_marker = 'tangible harm definitively occurred'

def classify_harm(val):
    if pd.isna(val):
        return 'Intangible' # Treat missing as Intangible/Unknown for this binary split or drop? 
        # Safer to treat as Intangible if we assume Tangible is the exception, 
        # but let's check if 'nan' means no info. 
        # For this experiment, let's map strictly based on the string.
    
    val_lower = str(val).lower()
    if tangible_marker in val_lower:
        return 'Tangible'
    else:
        return 'Intangible'

df_aiid['harm_type'] = df_aiid['Tangible Harm'].apply(classify_harm)

# --- Step 3: Analysis ---

# Create Contingency Table
contingency_table = pd.crosstab(df_aiid['is_genai'], df_aiid['harm_type'])

# Rename indices for clarity
contingency_table.index = ['Non-GenAI', 'GenAI']
print("\n--- Contingency Table (GenAI vs Harm Type) ---")
print(contingency_table)

# Check if we have data in both columns
if 'Tangible' not in contingency_table.columns:
    contingency_table['Tangible'] = 0
if 'Intangible' not in contingency_table.columns:
    contingency_table['Intangible'] = 0

# Calculate percentages for context
contingency_pct = pd.crosstab(df_aiid['is_genai'], df_aiid['harm_type'], normalize='index') * 100
print("\n--- Percentages (Row-wise) ---")
print(contingency_pct)

# Statistical Test
# Using Fisher's Exact Test if sample size is small, otherwise Chi2.
# Given the likely imbalance, Fisher's is safer or Chi2 with Yates correction.
# We will use Fisher's Exact Test for 2x2.

odds_ratio, p_value = stats.fisher_exact(contingency_table.loc[['Non-GenAI', 'GenAI'], ['Intangible', 'Tangible']])

print(f"\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
print("\n--- Interpretation ---")
if p_value < 0.05:
    print("Result: Statistically Significant.")
    if odds_ratio > 1:
        print("GenAI incidents are significantly more likely to be associated with Intangible harm (vs Tangible) compared to Non-GenAI.")
    else:
        print("GenAI incidents are significantly LESS likely to be associated with Intangible harm compared to Non-GenAI.")
else:
    print("Result: Not Statistically Significant. No evidence that GenAI harm profiles differ from Non-GenAI in this dataset.")
