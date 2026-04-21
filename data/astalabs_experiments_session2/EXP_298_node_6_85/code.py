import pandas as pd
import scipy.stats as stats
import sys

# Load dataset
print("Loading dataset...")
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
print(f"ATLAS cases loaded: {len(atlas_df)}")

# Define keywords
physical_keywords = [
    'automotive', 'car', 'vehicle', 'drive', 'driving', 'energy', 'power', 'grid', 
    'defense', 'military', 'weapon', 'healthcare', 'medical', 'hospital', 
    'surveillance', 'camera', 'cctv', 'drone', 'robot', 'biometric', 'physical', 
    'traffic', 'face', 'facial', 'recognition', 'sensor', 'gps'
]

digital_keywords = [
    'finance', 'bank', 'trading', 'software', 'malware', 'phishing', 'email', 
    'internet', 'web', 'cloud', 'network', 'chatbot', 'language model', 'llm', 
    'text', 'spam', 'bypass', 'antivirus', 'translation', 'bot', 'algorithm', 
    'filter', 'online', 'media'
]

# Function to categorize sector
def categorize_sector(row):
    # Combine name and summary, handle NaN
    text = str(row.get('name', '')) + " " + str(row.get('summary', ''))
    text = text.lower()
    
    # Check Physical first (prioritizing domain of application)
    for kw in physical_keywords:
        if kw in text:
            return 'Physical'
            
    # Check Digital
    for kw in digital_keywords:
        if kw in text:
            return 'Digital'
            
    return 'Unclassified'

# Apply categorization
atlas_df['inferred_sector'] = atlas_df.apply(categorize_sector, axis=1)

# Identify Impact tactic
# Tactics seem to be strings like "{{impact.id}}|..."
atlas_df['has_impact'] = atlas_df['tactics'].astype(str).str.contains('impact', case=False, na=False)

# Filter out Unclassified
analysis_df = atlas_df[atlas_df['inferred_sector'] != 'Unclassified'].copy()

print("\nSector Inference Results:")
print(analysis_df['inferred_sector'].value_counts())
print(f"Unclassified cases dropped: {len(atlas_df) - len(analysis_df)}")

# Create Contingency Table
contingency_table = pd.crosstab(analysis_df['inferred_sector'], analysis_df['has_impact'])
print("\nContingency Table (Sector vs Has Impact):")
print(contingency_table)

# Check if we have enough data for 2x2
if contingency_table.shape == (2, 2):
    # Perform Fisher's Exact Test
    # Table structure:
    #              False  True
    # inferred_sector
    # Digital       A      B
    # Physical      C      D
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    
    print(f"\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print("Result: Statistically significant association found.")
    else:
        print("Result: No statistically significant association found.")
        
    # Calculate percentages for clarity
    physical_total = contingency_table.loc['Physical'].sum()
    physical_impact = contingency_table.loc['Physical', True]
    digital_total = contingency_table.loc['Digital'].sum()
    digital_impact = contingency_table.loc['Digital', True]
    
    print(f"\nPhysical Sector Impact Rate: {physical_impact}/{physical_total} ({physical_impact/physical_total:.2%})")
    print(f"Digital Sector Impact Rate: {digital_impact}/{digital_total} ({digital_impact/digital_total:.2%})")

else:
    print("\nInsufficient data dimensions for Fisher's Exact Test (need 2x2 table).")
    print("Observed shape:", contingency_table.shape)
