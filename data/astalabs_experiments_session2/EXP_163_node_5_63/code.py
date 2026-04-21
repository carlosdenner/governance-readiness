import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 subset
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# Map Topics to Experiment Categories
# Based on previous output, we map the actual dataset values to the target labels
topic_mapping = {
    'Health & Medical': 'Health',
    'Law & Justice': 'Law Enforcement',
    'Government Services (includes Benefits and Service Delivery)': 'Government Services'
}

eo_df['topic_mapped'] = eo_df['8_topic_area'].map(topic_mapping)

# Filter for only the mapped topics
subset = eo_df.dropna(subset=['topic_mapped']).copy()

print(f"Filtered subset size: {len(subset)}")
print("Counts per topic in subset:")
print(subset['topic_mapped'].value_counts())

# Clean Mitigation Column (62_disparity_mitigation)
def clean_mitigation(val):
    if pd.isna(val):
        return 0
    
    val_str = str(val).lower().strip()
    
    # explicit negative indicators
    negative_starts = ['n/a', 'no ', 'none', 'not ', '0', 'false']
    if any(val_str.startswith(x) for x in negative_starts):
        return 0
    
    # specific negative phrases found in previous inspection
    if 'does not take into account' in val_str:
        return 0
    if 'not safety or rights-impacting' in val_str:
        return 0
        
    # If it contains content that isn't negative, assume it describes a mitigation
    return 1

subset['has_mitigation'] = subset['62_disparity_mitigation'].apply(clean_mitigation)

# Verify cleaning
print("\nMitigation distribution (Binary) by Topic:")
print(subset.groupby('topic_mapped')['has_mitigation'].value_counts().unstack())

# Contingency Table
contingency = pd.crosstab(subset['topic_mapped'], subset['has_mitigation'])
print("\nContingency Table:")
print(contingency)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}, p-value: {p:.4e}")

# Calculate Rates
rates = subset.groupby('topic_mapped')['has_mitigation'].mean()
print("\nMitigation Rates by Sector:")
print(rates)

# Visualization
plt.figure(figsize=(10, 6))
ax = rates.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Disparity Mitigation Reporting Rates by Topic Area')
plt.ylabel('Proportion Reporting Mitigation')
plt.xlabel('Topic Area')
plt.ylim(0, 1.0)
plt.axhline(y=rates.mean(), color='r', linestyle='--', label=f'Mean Rate ({rates.mean():.2f})')

# Add value labels
for i, v in enumerate(rates):
    ax.text(i, v + 0.02, f"{v:.1%}", ha='center')

plt.legend()
plt.tight_layout()
plt.show()