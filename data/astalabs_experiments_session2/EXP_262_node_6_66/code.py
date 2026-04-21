import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    # Reading with low_memory=False to handle mixed types warning from previous steps
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if file is in current directory
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

print(f"Total AIID incidents: {len(aiid_df)}")

# Identify relevant columns based on previous metadata
# 84_known_ai_technology and 78_sector_of_deployment
# Normalize column names to be safe
aiid_df.columns = [c.strip().lower().replace(' ', '_').replace(':', '') for c in aiid_df.columns]

# Find the specific columns
tech_col = next((c for c in aiid_df.columns if 'known_ai_technology' in c), None)
sector_col = next((c for c in aiid_df.columns if 'sector_of_deployment' in c), None)

if not tech_col or not sector_col:
    print("Could not identify required columns. Available columns:")
    print(aiid_df.columns.tolist())
    exit()

print(f"Using technology column: {tech_col}")
print(f"Using sector column: {sector_col}")

# Fill NaNs
aiid_df[tech_col] = aiid_df[tech_col].fillna('')
aiid_df[sector_col] = aiid_df[sector_col].fillna('')

# Define GenAI keywords
genai_keywords = ['generative', 'language model', 'llm', 'gpt', 'diffusion', 'chatbot', 'transformer', 'foundation model']

# Create is_genai flag
aiid_df['is_genai'] = aiid_df[tech_col].apply(lambda x: any(k in str(x).lower() for k in genai_keywords))

# Define Sector Groups
# Tech/Info vs Safety-Critical
# Let's inspect unique sectors first to ensure correct mapping
unique_sectors = aiid_df[sector_col].unique()
print(f"\nTop 10 Sectors found:\n{aiid_df[sector_col].value_counts().head(10)}")

def map_sector(sector_str):
    s = str(sector_str).lower()
    if any(x in s for x in ['technology', 'media', 'information', 'internet', 'software', 'telecom', 'entertainment']):
        return 'Tech/Info'
    elif any(x in s for x in ['healthcare', 'transportation', 'energy', 'automotive', 'aviation', 'medical', 'hospital', 'utility', 'defense', 'military']):
        return 'Safety-Critical'
    return 'Other'

aiid_df['sector_group'] = aiid_df[sector_col].apply(map_sector)

# Filter for only the two groups of interest
analysis_df = aiid_df[aiid_df['sector_group'].isin(['Tech/Info', 'Safety-Critical'])].copy()

print(f"\nAnalysis set size (filtered for relevant sectors): {len(analysis_df)}")

# Contingency Table
contingency_table = pd.crosstab(analysis_df['is_genai'], analysis_df['sector_group'])
print("\nContingency Table (Count):")
print(contingency_table)

# Calculate Proportions
prop_table = pd.crosstab(analysis_df['is_genai'], analysis_df['sector_group'], normalize='columns')
print("\nContingency Table (Proportions):")
print(prop_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"p-value: {p:.4e}")

# Interpretation
genai_tech_rate = prop_table.loc[True, 'Tech/Info'] if True in prop_table.index else 0
genai_safety_rate = prop_table.loc[True, 'Safety-Critical'] if True in prop_table.index else 0

print(f"\nGenAI Incidence Rate in Tech/Info: {genai_tech_rate:.2%}")
print(f"GenAI Incidence Rate in Safety-Critical: {genai_safety_rate:.2%}")

if p < 0.05:
    print("Result: Statistically Significant Difference.")
    if genai_tech_rate > genai_safety_rate:
        print("Hypothesis Supported: GenAI is more concentrated in Tech/Info sectors.")
    else:
        print("Hypothesis Refuted: GenAI is more concentrated in Safety-Critical sectors.")
else:
    print("Result: No Statistically Significant Difference found.")