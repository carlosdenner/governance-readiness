import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# [debug]
print("Starting experiment: The Generative Malice Gap (Attempt 2)")

# Load dataset
file_name = 'astalabs_discovery_all_data.csv'
paths = [f'../{file_name}', file_name]
ds_path = next((p for p in paths if os.path.exists(p)), None)

if not ds_path:
    print(f"Error: Dataset {file_name} not found.")
    exit(1)

df = pd.read_csv(ds_path, low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID incidents: {len(aiid_df)}")

# Find correct column names dynamically
cols = df.columns.tolist()
col_tech = next((c for c in cols if 'Known AI Technology' in str(c)), None)
col_intent = next((c for c in cols if 'Intentional Harm' in str(c)), None)

print(f"Identified Tech Column: {col_tech}")
print(f"Identified Intent Column: {col_intent}")

if not col_tech or not col_intent:
    print("Could not identify required columns. Available columns:")
    # Print columns that might be relevant or a sample
    print([c for c in cols if 'AI' in str(c) or 'Harm' in str(c)])
    exit(1)

# Define Generative keywords
gen_keywords = [
    'generative', 'llm', 'gpt', 'diffusion', 'gan', 'transformer', 
    'chatbot', 'language model', 'text-to-image', 'deepfake', 'synthetic media',
    'stable diffusion', 'midjourney', 'dall-e', 'bard', 'chatgpt', 'gemini'
]

def categorize_tech(val):
    if pd.isna(val):
        return 'Unknown'
    val_lower = str(val).lower()
    if any(k in val_lower for k in gen_keywords):
        return 'Generative'
    return 'Predictive/Other'

aiid_df['tech_category'] = aiid_df[col_tech].apply(categorize_tech)

# Define Intentionality mapping
def categorize_intent(val):
    if pd.isna(val):
        return 'Unclear'
    val_lower = str(val).lower()
    # Check for 'yes', 'true' for Intentional
    # Check for 'no', 'false' for Unintentional
    if val_lower in ['yes', 'true', 'intentional']:
        return 'Intentional'
    elif val_lower in ['no', 'false', 'unintentional', 'accidental']:
        return 'Unintentional'
    # Sometimes values are sentences, so simpler check:
    if 'yes' in val_lower or 'true' in val_lower:
        return 'Intentional'
    if 'no' in val_lower or 'false' in val_lower:
        return 'Unintentional'
    return 'Unclear'

aiid_df['intent_category'] = aiid_df[col_intent].apply(categorize_intent)

# Filter for analysis (exclude Unknown tech and Unclear intent)
analysis_df = aiid_df[
    (aiid_df['tech_category'] != 'Unknown') & 
    (aiid_df['intent_category'] != 'Unclear')
].copy()

print(f"\nData points for analysis: {len(analysis_df)}")
print("Distribution by Tech Category:")
print(analysis_df['tech_category'].value_counts())
print("Distribution by Intent Category:")
print(analysis_df['intent_category'].value_counts())

if len(analysis_df) < 5:
    print("Not enough data points for statistical analysis.")
else:
    # Contingency Table
    contingency = pd.crosstab(analysis_df['tech_category'], analysis_df['intent_category'])
    print("\nContingency Table:")
    print(contingency)

    # Proportions (Row-wise to see % Intentional per Tech Category)
    props = pd.crosstab(analysis_df['tech_category'], analysis_df['intent_category'], normalize='index')
    print("\nProportions (Row-normalized):")
    print(props)

    # Statistical Test (Chi-Square)
    # We are testing if there is an association between Tech Type and Intentionality
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Plotting
    # Reorder for visualization if needed
    # We want a stacked bar of Intentional vs Unintentional
    ax = props.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#ff9999', '#66b3ff'])
    plt.title('Intentionality of Harm by AI Technology Type')
    plt.ylabel('Proportion of Incidents')
    plt.xlabel('Technology Category')
    plt.xticks(rotation=0)
    plt.legend(title='Intentional Harm', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()
