import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Check file existence
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from {file_path}...")
df = pd.read_csv(file_path, low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)} rows")

# 1. Classify Technology
def classify_technology(text):
    if pd.isna(text):
        return None
    text = str(text).lower()
    genai_keywords = ['generative', 'llm', 'large language model', 'gpt', 'diffusion', 
                      'dall-e', 'midjourney', 'stable diffusion', 'transformer', 'chatbot', 
                      'foundation model', 'chatgpt', 'bert', 'palm', 'llama', 'gan', 'stylegan']
    
    if any(k in text for k in genai_keywords):
        return 'Generative AI'
    return 'Traditional AI'

aiid_df['Tech_Type'] = aiid_df['Known AI Technology'].apply(classify_technology)

# 2. Classify Harm
# Logic: 
# - Tangible if 'Tangible Harm' says it definitively occurred AND 'Special Interest Intangible Harm' is NOT yes.
# - Intangible if 'Special Interest Intangible Harm' says yes AND 'Tangible Harm' did NOT definitively occur.
# - Exclude mixed/ambiguous cases to strictly test the skew.

def classify_harm_composite(row):
    tangible_val = str(row.get('Tangible Harm', '')).lower()
    intangible_val = str(row.get('Special Interest Intangible Harm', '')).lower()
    
    is_tangible = 'tangible harm definitively occurred' in tangible_val
    is_intangible = 'yes' in intangible_val
    
    if is_tangible and not is_intangible:
        return 'Tangible'
    elif is_intangible and not is_tangible:
        return 'Intangible'
    else:
        return None # Mixed or None

aiid_df['Harm_Type'] = aiid_df.apply(classify_harm_composite, axis=1)

# 3. Filter Data for Analysis
analysis_df = aiid_df.dropna(subset=['Tech_Type', 'Harm_Type']).copy()

print(f"\nData points for analysis: {len(analysis_df)}")
print("Distribution by Tech Type:")
print(analysis_df['Tech_Type'].value_counts())
print("\nDistribution by Harm Type:")
print(analysis_df['Harm_Type'].value_counts())

# 4. Statistical Test
contingency_table = pd.crosstab(analysis_df['Tech_Type'], analysis_df['Harm_Type'])
print("\nContingency Table (Observed):")
print(contingency_table)

if not contingency_table.empty and contingency_table.size >= 4:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Interpret results
    alpha = 0.05
    if p < alpha:
        print("Result: Significant association between AI Technology and Harm Type.")
    else:
        print("Result: No significant association found.")

    # 5. Visualization
    # Normalize to get percentages for stacked bar to visualize the 'skew'
    ct_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    
    ax = ct_pct.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon'])
    plt.title('Proportion of Tangible vs Intangible Harms by AI Technology')
    plt.xlabel('AI Technology Type')
    plt.ylabel('Percentage')
    plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # Annotate bars
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
        
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient data for Chi-Square test.")
