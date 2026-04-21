import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

# --- Load Data ---
filename = 'astalabs_discovery_all_data.csv'
if not os.path.exists(filename):
    if os.path.exists('../' + filename):
        filename = '../' + filename

print(f"Loading dataset from: {filename}")
df = pd.read_csv(filename, low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)} records")

# --- 1. Define Tech Categories (Biometric vs Other) ---
# Correct Column Name found in debug: 'Known AI Technology'
tech_col = 'Known AI Technology'

# keywords for Biometrics/Facial Recognition
bio_keywords = ['face', 'facial', 'biometric', 'recognition', 'gait', 'iris', 'voice print']

def categorize_tech(text):
    if pd.isna(text):
        return 'Other'
    text_lower = str(text).lower()
    if any(k in text_lower for k in bio_keywords):
        return 'Biometric'
    return 'Other'

aiid_df['Tech_Category'] = aiid_df[tech_col].apply(categorize_tech)

# --- 2. Define Harm Categories (Civil Rights vs Other) ---
# Correct Column Name found in debug: 'Harm Domain'
# Mapping AIID harm domains to 'Civil Rights' broadly.
harm_col = 'Harm Domain'

civil_rights_keywords = ['civil rights', 'liberty', 'privacy', 'discrimination', 'allocative', 'representation']

def categorize_harm(text):
    if pd.isna(text):
        return 'Other'
    text_lower = str(text).lower()
    if any(k in text_lower for k in civil_rights_keywords):
        return 'Civil Rights'
    return 'Other'

aiid_df['Harm_Category'] = aiid_df[harm_col].apply(categorize_harm)

# --- 3. Analysis ---

# Generate Contingency Table
contingency_table = pd.crosstab(aiid_df['Tech_Category'], aiid_df['Harm_Category'])

print("\n--- Contingency Table (Counts) ---")
print(contingency_table)

# Calculate Percentages
# Row-wise normalization to see: Of Biometric techs, what % are Civil Rights harms?
row_pct = pd.crosstab(aiid_df['Tech_Category'], aiid_df['Harm_Category'], normalize='index') * 100
print("\n--- Contingency Table (Percentages by Tech) ---")
print(row_pct.round(2))

# Perform Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n--- Chi-Square Test Results ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4e}")

# Interpretation
alpha = 0.05
print("\n--- Conclusion ---")
if p < alpha:
    print("Result: Statistically Significant.")
    print("The data supports the hypothesis that Biometric technologies have a different harm profile (likely skewed towards Civil Rights/Liberties) compared to other AI technologies.")
else:
    print("Result: Not Statistically Significant.")
    print("The data does not show a significant association between Biometric technologies and Civil Rights harms compared to other technologies.")

# Additional context
print("\n[Debug Info] Top 5 Tech strings classified as Biometric:")
print(aiid_df[aiid_df['Tech_Category']=='Biometric'][tech_col].value_counts().head(5))

print("\n[Debug Info] Top 5 Harm Domains classified as Civil Rights:")
print(aiid_df[aiid_df['Harm_Category']=='Civil Rights'][harm_col].value_counts().head(5))
