import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Define dataset path
dataset_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from {dataset_path}...")
try:
    df = pd.read_csv(dataset_path, low_memory=False)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID incidents loaded: {len(aiid_df)}")

# 1. Normalize and Filter Sectors
sector_col = 'Sector of Deployment'

def map_sector(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    if any(x in val_str for x in ['health', 'medical', 'hospital', 'clinic', 'doctor', 'patient']):
        return 'Healthcare'
    elif any(x in val_str for x in ['financ', 'bank', 'trading', 'insurance', 'credit', 'loan', 'money']):
        return 'Financial'
    else:
        return 'Other'

aiid_df['normalized_sector'] = aiid_df[sector_col].apply(map_sector)

# 2. Categorize Harm using Text Analysis (Fallback strategy since structured columns were uninformative)
# We combine title and description for a richer context
aiid_df['text_content'] = aiid_df['title'].fillna('') + ' ' + aiid_df['description'].fillna('')

def map_harm_text(text):
    text = str(text).lower()
    
    # Keywords for Physical Harm (Safety, Life, Health)
    physical_keywords = ['death', 'dead', 'kill', 'injur', 'hurt', 'physical', 'safety', 'bodily', 
                         'crash', 'accident', 'burn', 'poison', 'patient harm']
    
    # Keywords for Economic Harm (Financial loss, Fraud, etc.)
    economic_keywords = ['financial', 'money', 'dollar', 'economic', 'loss', 'credit', 'bank', 
                         'fraud', 'scam', 'theft', 'fund', 'wallet', 'crypto', 'payment', 'charge']
    
    has_physical = any(k in text for k in physical_keywords)
    has_economic = any(k in text for k in economic_keywords)
    
    if has_physical and not has_economic:
        return 'Physical'
    elif has_economic and not has_physical:
        return 'Economic'
    elif has_physical and has_economic:
        # Conflict resolution: usually physical takes precedence in severity, 
        # but for this study let's call it 'Mixed/Physical'
        return 'Physical'
    else:
        return 'Other'

aiid_df['harm_category'] = aiid_df['text_content'].apply(map_harm_text)

# Filter only for the target sectors
study_df = aiid_df[aiid_df['normalized_sector'].isin(['Healthcare', 'Financial'])].copy()
print(f"Incidents in target sectors (Healthcare/Financial): {len(study_df)}")

# 3. Create Contingency Table
contingency_table = pd.crosstab(study_df['normalized_sector'], study_df['harm_category'])

# Ensure columns exist
for col in ['Physical', 'Economic', 'Other']:
    if col not in contingency_table.columns:
        contingency_table[col] = 0

# Reorder
contingency_table = contingency_table[['Physical', 'Economic', 'Other']]

print("\n--- Contingency Table (Sector vs Harm) ---")
print(contingency_table)

# 4. Perform Chi-Square Test
# We check if we have enough data
if len(study_df) < 5:
    print("\nWarning: Sample size too small for reliable Chi-Square test.")
else:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Degrees of Freedom: {dof}")

    alpha = 0.05
    if p < alpha:
        print("Result: Statistically Significant (Reject Null Hypothesis)")
    else:
        print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")

# 5. Calculate Row Percentages
row_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
print("\n--- Row Percentages ---")
print(row_pct.round(2))

# 6. Visualize
try:
    plt.figure(figsize=(10, 6))
    ax = row_pct.plot(kind='bar', stacked=True, colormap='RdYlBu', figsize=(10, 6))
    plt.title('Distribution of Harm Types by Sector (Text-Inferred)')
    plt.xlabel('Sector')
    plt.ylabel('Percentage of Incidents')
    plt.legend(title='Harm Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Add labels if possible
    for c in ax.containers:
        # Only label if segment is big enough
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 5 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center')
        
    plt.show()
except Exception as e:
    print(f"Plotting error: {e}")
