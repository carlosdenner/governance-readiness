import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load Dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    # Try loading from current directory
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Try loading from parent directory as per instructions
    df = pd.read_csv('../' + file_path, low_memory=False)

print("Dataset loaded.")

# 2. Filter for AIID Incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents: {len(aiid_df)}")

# 3. Identify Columns
# Find 'Known AI Technology' column
tech_cols = [c for c in aiid_df.columns if 'Known AI Technology' in str(c)]
if not tech_cols:
    # Fallback
    tech_cols = [c for c in aiid_df.columns if 'Technology' in str(c)]
tech_col = tech_cols[0]

# Find 'Tangible Harm' column
# Note: 'Harm Domain' was found to be binary in previous steps. 
# We target 'Tangible Harm' which contains specific harm tags.
harm_cols = [c for c in aiid_df.columns if 'Tangible Harm' in str(c)]
if not harm_cols:
    # Fallback to general harm search excluding known binary/irrelevant columns
    harm_cols = [c for c in aiid_df.columns if 'harm' in str(c).lower() 
                 and 'domain' not in str(c).lower() 
                 and 'level' not in str(c).lower()
                 and 'basis' not in str(c).lower()]
harm_col = harm_cols[0]

print(f"Using Tech Column: {tech_col}")
print(f"Using Harm Column: {harm_col}")

# Debug: Print sample values to confirm we have the right column
print(f"Sample values in {harm_col}:", aiid_df[harm_col].dropna().astype(str).unique()[:10])

# 4. Create GenAI Flag
genai_keywords = [
    'generative', 'llm', 'gpt', 'diffusion', 'chatbot', 'large language model',
    'transformer', 'dall-e', 'midjourney', 'stable diffusion', 'bard', 'gemini', 
    'llama', 'copilot', 'chatgpt', 'gan', 'foundation model'
]

aiid_df[tech_col] = aiid_df[tech_col].fillna('').astype(str)
aiid_df['is_genai'] = aiid_df[tech_col].apply(
    lambda x: 'Generative AI' if any(k in x.lower() for k in genai_keywords) else 'Discriminative/Other'
)

# 5. Map Harm Categories
def map_harm(val):
    s = str(val).lower()
    # Allocative Mappings
    # 'financial', 'economic', 'property', 'professional' are standard tags for allocative harm
    if any(x in s for x in ['financial', 'economic', 'property', 'professional', 'hiring', 'employment', 'allocative']):
        return 'Allocative'
    # Societal/Reputational Mappings
    # 'reputation', 'psychological', 'civil rights', 'social' are standard tags
    if any(x in s for x in ['reputation', 'psychological', 'civil rights', 'social', 'discrimination', 'privacy', 'civil liberties', 'representation']):
        return 'Societal/Reputational'
    return 'Other'

aiid_df['Harm_Category'] = aiid_df[harm_col].apply(map_harm)

# 6. Filter for Analysis
analysis_df = aiid_df[aiid_df['Harm_Category'].isin(['Allocative', 'Societal/Reputational'])].copy()
print(f"\nIncidents remaining after filtering for Allocative/Societal harms: {len(analysis_df)}")
print(analysis_df['Harm_Category'].value_counts())

# 7. Statistics & Visualization
if len(analysis_df) > 0:
    # Contingency Table
    ct = pd.crosstab(analysis_df['is_genai'], analysis_df['Harm_Category'])
    print("\nContingency Table:")
    print(ct)
    
    # Percentages
    ct_pct = pd.crosstab(analysis_df['is_genai'], analysis_df['Harm_Category'], normalize='index') * 100
    print("\nPercentage Distribution:")
    print(ct_pct)
    
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"\nChi-Square Results: Statistic={chi2:.4f}, p-value={p:.4f}")
    
    # Interpretation
    alpha = 0.05
    if p < alpha:
        print("\nResult: Statistically significant difference found.")
        # Check direction
        gen_soc_rate = ct_pct.loc['Generative AI', 'Societal/Reputational'] if 'Generative AI' in ct_pct.index else 0
        disc_soc_rate = ct_pct.loc['Discriminative/Other', 'Societal/Reputational'] if 'Discriminative/Other' in ct_pct.index else 0
        
        print(f"GenAI Societal Rate: {gen_soc_rate:.2f}%")
        print(f"Discriminative Societal Rate: {disc_soc_rate:.2f}%")
        
        if gen_soc_rate > disc_soc_rate:
            print("Hypothesis Supported: GenAI is more associated with Societal/Reputational harms.")
        else:
            print("Hypothesis Refuted: GenAI is LESS associated with Societal/Reputational harms.")
    else:
        print("\nResult: No statistically significant difference found (p >= 0.05).")
    
    # Plot
    plt.figure(figsize=(10, 6))
    ct_pct.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], ax=plt.gca())
    plt.title('Harm Distribution: Generative vs Discriminative AI')
    plt.ylabel('Percentage')
    plt.xlabel('AI System Type')
    plt.legend(title='Harm Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

else:
    print("No data available for analysis after filtering. Check harm column values.")
