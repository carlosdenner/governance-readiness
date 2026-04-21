import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Load dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# Clean Autonomy Level
autonomy_map = {
    'Autonomy1': 1, 
    'Autonomy2': 2, 
    'Autonomy3': 3
}
aiid['autonomy_score'] = aiid['Autonomy Level'].map(autonomy_map)

# Clean Sector of Deployment
def map_domain(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    
    physical_keywords = ['transport', 'health', 'medic', 'manufactur', 'industr', 
                         'energy', 'agricultur', 'construct', 'robot']
    digital_keywords = ['financ', 'bank', 'educat', 'govern', 'public', 'media', 
                        'entertain', 'retail', 'consum', 'service']
    
    # Check physical first (arbitrary priority, or could be exclusive)
    if any(k in val_str for k in physical_keywords):
        return 'Physical'
    elif any(k in val_str for k in digital_keywords):
        return 'Digital'
    return None

aiid['risk_domain'] = aiid['Sector of Deployment'].apply(map_domain)

# Drop rows with missing values for the analysis
analysis_df = aiid.dropna(subset=['autonomy_score', 'risk_domain'])

# Descriptive Statistics
print("--- Analysis Counts ---")
counts = analysis_df['risk_domain'].value_counts()
print(counts)

# Statistical Test
physical_scores = analysis_df[analysis_df['risk_domain'] == 'Physical']['autonomy_score']
digital_scores = analysis_df[analysis_df['risk_domain'] == 'Digital']['autonomy_score']

print("\n--- Mann-Whitney U Test ---")
if len(physical_scores) > 0 and len(digital_scores) > 0:
    stat, p = mannwhitneyu(physical_scores, digital_scores, alternative='two-sided')
    print(f"U-statistic: {stat}")
    print(f"p-value: {p:.5f}")
    if p < 0.05:
        print("Result: Significant difference in autonomy levels between Physical and Digital domains.")
    else:
        print("Result: No significant difference found.")
        
    # Calculate medians for context
    print(f"Median Autonomy (Physical): {physical_scores.median()}")
    print(f"Median Autonomy (Digital): {digital_scores.median()}")
else:
    print("Insufficient data for statistical testing.")

# Visualization
plt.figure(figsize=(8, 6))
sns.boxplot(x='risk_domain', y='autonomy_score', data=analysis_df, order=['Physical', 'Digital'])
plt.title('AI Autonomy Levels: Physical vs Digital Sectors')
plt.ylabel('Autonomy Level (1=Low, 3=High)')
plt.xlabel('Risk Domain')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()