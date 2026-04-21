import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Define the file path
file_path = 'astalabs_discovery_all_data.csv'

try:
    # Load dataset
    df = pd.read_csv(file_path, low_memory=False)
    
    # Filter for AIID incidents
    aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"Loaded AIID incidents: {len(aiid)} rows")
    
    # --- Preprocessing Sectors ---
    # Ensure Sector of Deployment is string, handle NaNs
    aiid['Sector of Deployment'] = aiid['Sector of Deployment'].fillna('').astype(str)
    
    # Split comma-separated sectors and explode
    aiid['Sector of Deployment'] = aiid['Sector of Deployment'].str.split(',')
    aiid = aiid.explode('Sector of Deployment')
    aiid.reset_index(drop=True, inplace=True)
    
    # Clean whitespace and filter empty/nan strings
    aiid['Sector of Deployment'] = aiid['Sector of Deployment'].str.strip()
    aiid = aiid[~aiid['Sector of Deployment'].isin(['', 'nan'])]
    
    # --- Define Harm Classification Logic ---
    def classify_harm(row):
        harms = []
        # Convert to string and lower case for robust matching
        tangible = str(row['Tangible Harm']).lower() if pd.notna(row['Tangible Harm']) else ''
        intangible = str(row['Special Interest Intangible Harm']).lower() if pd.notna(row['Special Interest Intangible Harm']) else ''
        
        # Physical Logic: Based on prompt instructions
        # 'definitively occurred' or 'imminent risk' in Tangible Harm
        # Also checking for explicit 'physical' to be safe given dataset nature
        if 'definitively occurred' in tangible or 'imminent risk' in tangible or 'physical' in tangible:
            harms.append('Physical')
            
        # Psychological Logic: Based on prompt instructions
        # 'yes' in Special Interest Intangible Harm
        # Also checking for explicit 'psychological' in Tangible Harm as a fallback/augmentation
        if 'yes' in intangible or 'psychological' in tangible:
            harms.append('Psychological')
            
        return harms

    # Apply classification
    aiid['Harm_Type'] = aiid.apply(classify_harm, axis=1)
    
    # Explode Harm_Type to handle cases with multiple harms
    aiid = aiid.explode('Harm_Type')
    aiid.reset_index(drop=True, inplace=True)
    
    # Filter for relevant harm types (remove rows that didn't match either)
    relevant_harms = aiid[aiid['Harm_Type'].isin(['Physical', 'Psychological'])]
    
    if relevant_harms.empty:
        print("No records found matching 'Physical' or 'Psychological' criteria.")
    else:
        # --- Analysis ---
        # Create Crosstab (Contingency Table)
        ct = pd.crosstab(relevant_harms['Harm_Type'], relevant_harms['Sector of Deployment'])
        
        # Calculate Probabilities (Row-wise normalization)
        probs = ct.div(ct.sum(axis=1), axis=0)
        
        # Calculate Shannon Entropy
        def calculate_entropy(p):
            p = p[p > 0] # Filter zero probabilities to avoid log(0)
            return -np.sum(p * np.log2(p))
        
        entropy_scores = probs.apply(calculate_entropy, axis=1)
        
        print("\n--- Entropy Scores (Higher = More Distributed) ---")
        print(entropy_scores)
        
        # Get top 5 sectors for each harm type
        print("\n--- Top 5 Sectors by Harm Type ---")
        top_sectors_dict = {}
        for harm in ['Physical', 'Psychological']:
            if harm in probs.index:
                top_5 = probs.loc[harm].sort_values(ascending=False).head(5)
                print(f"\n{harm} Harm Top Sectors:")
                print(top_5)
                top_sectors_dict[harm] = top_5.index.tolist()
        
        # --- Visualization ---
        # Collect all unique top sectors to display in the chart
        all_top_sectors = set()
        for sectors in top_sectors_dict.values():
            all_top_sectors.update(sectors)
        
        # Filter probabilities to only these top sectors for a cleaner chart
        plot_data = probs[list(all_top_sectors)].T
        
        # Sort for better visualization (optional)
        plot_data = plot_data.sort_index()
        
        ax = plot_data.plot(kind='bar', figsize=(12, 7), width=0.8)
        plt.title('Sector Distribution of Physical vs. Psychological Harms (Top Sectors)')
        plt.ylabel('Proportion of Incidents')
        plt.xlabel('Sector')
        plt.legend(title='Harm Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Print raw counts for verification
        print("\n--- Raw Counts for Validation ---")
        print(ct[list(all_top_sectors)])

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the working directory.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
