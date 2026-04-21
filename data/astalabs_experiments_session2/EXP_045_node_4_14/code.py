import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

print(f"Loaded {len(aiid_df)} rows from AIID incidents.")

# Define columns
autonomy_col = 'Autonomy Level'
harm_col = 'AI Harm Level'

# Print unique values for verification
print("Unique Autonomy values:", aiid_df[autonomy_col].unique())
print("Unique Harm values:", aiid_df[harm_col].unique())

# Define mappings based on observed values
autonomy_mapping = {
    'Autonomy1': 1,  # System provides information/assists
    'Autonomy2': 2,  # System selects action/human in loop
    'Autonomy3': 3   # System acts/autonomous
}

# Mapping logic: None < Near-miss < Issue < Event
harm_mapping = {
    'none': 0,
    'AI tangible harm near-miss': 1,
    'AI tangible harm issue': 2,
    'AI tangible harm event': 3
}

# Apply mappings
aiid_df['Autonomy_Ordinal'] = aiid_df[autonomy_col].map(autonomy_mapping)
aiid_df['Harm_Ordinal'] = aiid_df[harm_col].map(harm_mapping)

# Drop NaNs (including 'unclear' or other unmapped values)
analysis_df = aiid_df.dropna(subset=['Autonomy_Ordinal', 'Harm_Ordinal'])

print(f"\nData points available for analysis after cleaning: {len(analysis_df)}")

if len(analysis_df) > 5:
    # Spearman Correlation
    corr, p_value = spearmanr(analysis_df['Autonomy_Ordinal'], analysis_df['Harm_Ordinal'])
    print(f"\nSpearman Correlation: {corr:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("Result: Statistically significant correlation.")
    else:
        print("Result: Not statistically significant.")

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Autonomy_Ordinal', y='Harm_Ordinal', data=analysis_df)
    plt.title('Harm Realization by Autonomy Level')
    plt.xlabel('Autonomy Level (1=Assist, 2=Select, 3=Act)')
    plt.ylabel('Harm Level (0=None, 1=Near-miss, 2=Issue, 3=Event)')
    
    # Set x-tick labels manually for clarity
    plt.xticks(ticks=[0, 1, 2], labels=['1: Assist', '2: Select', '3: Act'])
    plt.yticks(ticks=[0, 1, 2, 3], labels=['None', 'Near-miss', 'Issue', 'Event'])
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # Crosstab for detailed counts
    print("\nContingency Table:")
    print(pd.crosstab(analysis_df['Autonomy_Ordinal'], analysis_df['Harm_Ordinal']))
else:
    print("Insufficient data points to perform correlation analysis.")