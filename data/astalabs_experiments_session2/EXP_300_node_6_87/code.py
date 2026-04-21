import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import re

def run_experiment():
    # Attempt to locate the dataset
    filename = 'astalabs_discovery_all_data.csv'
    if not os.path.exists(filename):
        # Fallback to parent directory as per instruction hint, though previous debug failed there.
        # This covers both cases dynamically.
        filename = '../astalabs_discovery_all_data.csv'
    
    if not os.path.exists(filename):
        print("Error: Dataset file 'astalabs_discovery_all_data.csv' not found in current or parent directory.")
        return

    # Load the dataset
    print(f"Loading dataset from: {filename}")
    try:
        df = pd.read_csv(filename, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Filter for AIID incidents
    aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"Loaded {len(aiid_df)} AIID incidents.")

    # Define column names based on metadata
    autonomy_col = 'Autonomy Level'
    harm_col = 'Harm Domain'

    # Verify columns exist
    if autonomy_col not in aiid_df.columns or harm_col not in aiid_df.columns:
        print(f"Required columns '{autonomy_col}' or '{harm_col}' missing.")
        print(f"Available columns: {aiid_df.columns.tolist()}")
        return

    # 1. Map Autonomy Level
    # High (Level 3-5) and Low (Level 0-2)
    def map_autonomy(val):
        val_str = str(val).lower().strip()
        if val_str == 'nan' or val_str == '':
            return np.nan
        
        # Extract digits
        digits = re.findall(r'\d+', val_str)
        if digits:
            level = int(digits[0])
            if 0 <= level <= 2:
                return 'Low Autonomy'
            elif level >= 3:
                return 'High Autonomy'
        
        # Fallback for text descriptions if no digits found
        if 'low' in val_str or 'no' in val_str:
            return 'Low Autonomy'
        if 'high' in val_str or 'full' in val_str:
            return 'High Autonomy'
            
        return np.nan

    # 2. Map Harm Domain
    # Physical vs Intangible/Economic
    def map_harm(val):
        val_str = str(val).lower().strip()
        if val_str == 'nan' or val_str == '':
            return np.nan
        
        # Keywords for physical harm
        physical_keywords = ['physical', 'safety', 'life', 'death', 'injury', 'bodily', 'violence', 'kill']
        if any(k in val_str for k in physical_keywords):
            return 'Physical'
        
        # Default to Intangible/Economic for everything else (e.g., discrimination, economic, reputation)
        return 'Intangible/Economic'

    # Apply mappings
    aiid_df['Autonomy_Class'] = aiid_df[autonomy_col].apply(map_autonomy)
    aiid_df['Harm_Class'] = aiid_df[harm_col].apply(map_harm)

    # Drop rows with missing values in relevant columns
    valid_df = aiid_df.dropna(subset=['Autonomy_Class', 'Harm_Class'])
    print(f"Valid data points for analysis: {len(valid_df)}")

    if len(valid_df) < 5:
        print("Insufficient data points for statistical analysis.")
        return

    # Generate Cross-tabulation
    contingency = pd.crosstab(valid_df['Autonomy_Class'], valid_df['Harm_Class'])
    print("\nContingency Table (Counts):")
    print(contingency)

    # Calculate Proportions
    props = pd.crosstab(valid_df['Autonomy_Class'], valid_df['Harm_Class'], normalize='index')
    print("\nContingency Table (Proportions):")
    print(props)

    # 3. Perform Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # 4. Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot stacked bar chart
    # Colors: Intangible (Grey/Blue), Physical (Red)
    # Note: Column order is alphabetical: 'Intangible/Economic', 'Physical'
    colors = ['#1f77b4', '#d62728'] 
    props.plot(kind='bar', stacked=True, ax=ax, color=colors, alpha=0.85)
    
    plt.title('Distribution of Harm Type by Autonomy Level (AIID)')
    plt.ylabel('Proportion of Incidents')
    plt.xlabel('Autonomy Level')
    plt.legend(title='Harm Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)

    # Annotate bars
    for n, x in enumerate([*props.index.values]):
        for (proportion, y_loc) in zip(props.loc[x], props.loc[x].cumsum()):
            # Label if segment is large enough
            if proportion > 0.05:
                label_text = f"{proportion*100:.1f}%"
                plt.text(x=n, y=(y_loc - proportion) + (proportion / 2),
                         s=label_text, 
                         color="white", fontsize=10, fontweight="bold", ha="center")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
