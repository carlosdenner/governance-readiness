import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt

# [debug]
print("Loading dataset...")

try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid)}")

# Identify relevant columns dynamically due to potential prefixes (e.g., '84: Known AI Technology')
tech_cols = [c for c in aiid.columns if 'Known AI Technology' in str(c)]
harm_cols = [c for c in aiid.columns if 'Harm Domain' in str(c)]
desc_cols = [c for c in aiid.columns if 'description' in str(c).lower() or 'summary' in str(c).lower()]

tech_col = tech_cols[0] if tech_cols else None
harm_col = harm_cols[0] if harm_cols else None
desc_col = desc_cols[0] if desc_cols else None

print(f"Using columns -> Tech: {tech_col}, Harm: {harm_col}, Description: {desc_col}")

if not tech_col or not harm_col:
    print("Error: Critical columns 'Known AI Technology' or 'Harm Domain' not found.")
else:
    # Define Keywords
    biometric_keywords = ['face', 'facial', 'biometric', 'surveillance']
    civil_rights_keywords = ['civil rights', 'discrimination', 'privacy', 'due process']

    def check_keywords(text, keywords):
        if pd.isna(text):
            return False
        text = str(text).lower()
        return any(k in text for k in keywords)

    # 1. Feature Engineering: Tech_Type
    # Check technology column
    aiid['is_biometric'] = aiid[tech_col].apply(lambda x: check_keywords(x, biometric_keywords))
    # Check description column if it exists
    if desc_col:
        aiid['is_biometric'] = aiid['is_biometric'] | aiid[desc_col].apply(lambda x: check_keywords(x, biometric_keywords))
    
    aiid['Tech_Type'] = np.where(aiid['is_biometric'], 'Biometric/Facial', 'Other')

    # 2. Feature Engineering: Harm_Type
    # Check harm column
    aiid['is_civil_rights'] = aiid[harm_col].apply(lambda x: check_keywords(x, civil_rights_keywords))
    # Check description column if it exists
    if desc_col:
        aiid['is_civil_rights'] = aiid['is_civil_rights'] | aiid[desc_col].apply(lambda x: check_keywords(x, civil_rights_keywords))
    
    aiid['Harm_Type'] = np.where(aiid['is_civil_rights'], 'Civil Rights', 'Other')

    # 3. Statistical Analysis
    # Create Contingency Table
    contingency = pd.crosstab(aiid['Tech_Type'], aiid['Harm_Type'])
    print("\n--- Contingency Table ---")
    print(contingency)

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")

    # Calculate Odds Ratio
    # Format:
    #               Civil Rights   Other
    # Biometric     a              b
    # Other         c              d
    if 'Civil Rights' in contingency.columns and 'Biometric/Facial' in contingency.index:
        a = contingency.loc['Biometric/Facial', 'Civil Rights']
        b = contingency.loc['Biometric/Facial', 'Other']
        c = contingency.loc['Other', 'Civil Rights']
        d = contingency.loc['Other', 'Other']
        
        try:
            odds_ratio = (a * d) / (b * c)
            print(f"Odds Ratio: {odds_ratio:.4f}")
        except ZeroDivisionError:
            print("Odds Ratio: Undefined (division by zero)")
    else:
        print("Odds Ratio could not be calculated due to missing categories.")

    # Visualizing
    # Normalize to get percentages for better comparison
    contingency_pct = contingency.div(contingency.sum(1), axis=0) * 100
    
    ax = contingency_pct.plot(kind='bar', stacked=True, color=['#d62728', '#1f77b4'], alpha=0.8)
    plt.title('Harm Distribution: Biometric vs Other Technologies')
    plt.ylabel('Percentage')
    plt.xlabel('Technology Type')
    plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
