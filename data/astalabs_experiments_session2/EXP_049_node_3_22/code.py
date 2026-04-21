import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def run_experiment():
    print("Starting Experiment: Regulatory Safety Buffers Analysis")
    
    # 1. Load Dataset
    # Try current directory first, then parent directory
    file_name = 'astalabs_discovery_all_data.csv'
    if os.path.exists(file_name):
        file_path = file_name
    elif os.path.exists(f'../{file_name}'):
        file_path = f'../{file_name}'
    else:
        print(f"Error: {file_name} not found in current or parent directory.")
        return

    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    # 2. Filter for AIID Incidents
    aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"Loaded {len(aiid_df)} AIID incidents.")

    # 3. Clean and Prepare Data
    # Columns of interest
    sector_col = 'Sector of Deployment'
    tangible_harm_col = 'Tangible Harm'
    harm_level_col = 'AI Harm Level'

    # Fill NAs
    aiid_df[sector_col] = aiid_df[sector_col].fillna('Unknown')
    aiid_df[tangible_harm_col] = aiid_df[tangible_harm_col].fillna('Unknown')
    aiid_df[harm_level_col] = aiid_df[harm_level_col].fillna('Unknown')

    # Deliverable 1: List unique values in Tangible Harm
    unique_tangible_harms = aiid_df[tangible_harm_col].unique()
    print(f"\nUnique values in '{tangible_harm_col}' (first 10):")
    print(unique_tangible_harms[:10])

    # 4. Classify Harm Status (Near-Miss vs Actual)
    # Logic: Check both Tangible Harm and AI Harm Level for 'near-miss' indicators
    def classify_harm(row):
        text = (str(row[tangible_harm_col]) + " " + str(row[harm_level_col])).lower()
        if any(x in text for x in ['near-miss', 'near miss', 'risk', 'potential', 'threat', 'unsafe', 'no harm']):
            return 'Near-Miss'
        else:
            return 'Actual Harm'

    aiid_df['Harm_Status'] = aiid_df.apply(classify_harm, axis=1)
    
    print("\nHarm Status Counts:")
    print(aiid_df['Harm_Status'].value_counts())

    # 5. Explode Sectors
    # Reset index to avoid duplicates after explode
    aiid_df[sector_col] = aiid_df[sector_col].astype(str).str.split(',')
    exploded_df = aiid_df.explode(sector_col).reset_index(drop=True)
    exploded_df[sector_col] = exploded_df[sector_col].str.strip()

    # 6. Define Regulation Tiers
    high_reg = ['Healthcare', 'Transportation', 'Energy', 'Financial', 'Finance', 'Health', 'Automotive', 'Aviation', 'Defense', 'Military', 'Government', 'Public Sector']
    low_reg = ['Entertainment', 'Retail', 'Social Media', 'Technology', 'Education', 'Consumer', 'Media', 'Other']

    def get_tier(sector):
        s_lower = sector.lower()
        for h in high_reg:
            if h.lower() in s_lower:
                return 'High'
        for l in low_reg:
            if l.lower() in s_lower:
                return 'Low'
        return 'Other'

    exploded_df['Regulation_Tier'] = exploded_df[sector_col].apply(get_tier)
    
    # Filter for High/Low only
    analysis_df = exploded_df[exploded_df['Regulation_Tier'].isin(['High', 'Low'])]

    # 7. Calculate Safety Buffer Ratio per Sector
    # Group by Sector and Tier
    sector_stats = analysis_df.groupby(['Sector of Deployment', 'Regulation_Tier', 'Harm_Status']).size().unstack(fill_value=0)
    
    if 'Near-Miss' not in sector_stats.columns:
        sector_stats['Near-Miss'] = 0
    if 'Actual Harm' not in sector_stats.columns:
        sector_stats['Actual Harm'] = 0
    
    # Filter sectors with very few incidents to avoid noise
    sector_stats['Total'] = sector_stats['Near-Miss'] + sector_stats['Actual Harm']
    sector_stats = sector_stats[sector_stats['Total'] >= 5]

    # Ratio: Near-Miss / Actual Harm
    # Add 1 to denominator to avoid division by zero (smoothing)
    sector_stats['Ratio'] = sector_stats['Near-Miss'] / (sector_stats['Actual Harm'] + 1)
    
    sector_stats = sector_stats.reset_index()
    
    print("\n--- Sector Analysis (Sample) ---")
    print(sector_stats.head())

    # 8. Statistical Comparison (Mann-Whitney U)
    high_ratios = sector_stats[sector_stats['Regulation_Tier'] == 'High']['Ratio']
    low_ratios = sector_stats[sector_stats['Regulation_Tier'] == 'Low']['Ratio']

    print(f"\nHigh Reg Sectors (n={len(high_ratios)}) Mean Ratio: {high_ratios.mean():.4f}")
    print(f"Low Reg Sectors (n={len(low_ratios)}) Mean Ratio: {low_ratios.mean():.4f}")

    if len(high_ratios) > 1 and len(low_ratios) > 1:
        u_stat, p_val = stats.mannwhitneyu(high_ratios, low_ratios, alternative='two-sided')
        print(f"\nMann-Whitney U Test: U={u_stat}, p={p_val:.4f}")
        if p_val < 0.05:
            print("Result: Significant difference.")
        else:
            print("Result: No significant difference.")
    else:
        print("Not enough data for statistical test.")

    # 9. Visualization
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Regulation_Tier', y='Ratio', data=sector_stats, palette='Set2')
    plt.title('Safety Buffer Ratio (Near-Miss / Actual Harm) by Regulation Tier')
    plt.ylabel('Ratio (Near-Miss / Actual Harm)')
    plt.xlabel('Regulation Tier')
    plt.show()

if __name__ == "__main__":
    run_experiment()