import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os
import sys

# --- Load Dataset ---
filename = 'astalabs_discovery_all_data.csv'
file_path = filename
if not os.path.exists(file_path):
    if os.path.exists(f'../{filename}'):
        file_path = f'../{filename}'

print(f"Loading dataset from: {file_path}")
try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Failed to load dataset: {e}")
    sys.exit(1)

# --- Filter for AIID Incidents ---
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)}")

# --- Column Identification ---
# Sector Column
sector_col = next((c for c in aiid_df.columns if 'sector' in c.lower() and 'deployment' in c.lower()), None)
if not sector_col:
    # Fallback search
    sector_col = next((c for c in aiid_df.columns if 'sector' in c.lower()), None)

# Description Column (for text analysis fallback)
text_col = next((c for c in aiid_df.columns if c.lower() in ['description', 'summary', 'text', 'content']), None)
if not text_col:
    # Try looking for long text columns
    for c in aiid_df.columns:
        if aiid_df[c].dtype == object and aiid_df[c].str.len().mean() > 50:
            text_col = c
            break

print(f"Using Sector Column: {sector_col}")
print(f"Using Text Column for Harm Classification: {text_col}")

if not sector_col or not text_col:
    print("Critical columns missing. Cannot proceed.")
    # Do not use exit(), just stop processing
else:
    # --- Classification Logic ---
    
    # Sector Mapping
    def map_sector(x):
        if pd.isna(x): return 'Unknown'
        x = str(x).lower()
        if 'fina' in x or 'bank' in x or 'insur' in x or 'credit' in x or 'trad' in x: return 'Financial'
        if 'gov' in x or 'public' in x or 'police' in x or 'justi' in x or 'law' in x or 'milit' in x or 'admin' in x: return 'Public Sector'
        if 'health' in x or 'medi' in x or 'hosp' in x: return 'Healthcare'
        if 'tech' in x or 'softw' in x or 'internet' in x or 'social media' in x: return 'Technology'
        if 'transport' in x or 'auto' in x or 'vehicle' in x: return 'Transportation'
        return 'Other'

    # Harm Mapping (Text Analysis)
    def derive_harm(text):
        if pd.isna(text): return 'Unknown'
        text = str(text).lower()
        
        # Keywords
        # Economic
        econ_kw = ['money', 'financial', 'fraud', 'theft', 'credit', 'bank', 'cost', 'price', 'market', 'economic', 'property', 'employment', 'job', 'hiring']
        # Physical
        phys_kw = ['death', 'dead', 'kill', 'injur', 'hurt', 'physical', 'safety', 'crash', 'accident', 'collision', 'medical', 'patient', 'health', 'burn']
        # Social/Civil
        soc_kw = ['bias', 'discriminat', 'racis', 'sexis', 'gender', 'ethnic', 'surveillance', 'privacy', 'arrest', 'police', 'jail', 'prison', 'rights', 'reputation', 'stereotyp', 'wrongful']
        
        # Scoring (simple presence check, priority: Physical > Social > Economic for overlapping cases, or strictly count)
        has_phys = any(k in text for k in phys_kw)
        has_soc = any(k in text for k in soc_kw)
        has_econ = any(k in text for k in econ_kw)
        
        if has_phys: return 'Physical'
        if has_soc: return 'Social/Civil'
        if has_econ: return 'Economic'
        return 'Other'

    # Apply Mappings
    aiid_df['mapped_sector'] = aiid_df[sector_col].apply(map_sector)
    aiid_df['derived_harm'] = aiid_df[text_col].apply(derive_harm)

    # --- Filter for Hypothesis Testing ---
    target_sectors = ['Financial', 'Public Sector']
    # target_sectors = ['Financial', 'Public Sector', 'Healthcare', 'Technology', 'Transportation'] # Extended for context
    target_harms = ['Economic', 'Social/Civil']
    # target_harms = ['Economic', 'Physical', 'Social/Civil'] # Extended for context
    
    # We keep the extended set for the plot to provide context, but focus metrics on the hypothesis
    plot_sectors = ['Financial', 'Public Sector', 'Healthcare', 'Technology', 'Transportation']
    plot_harms = ['Economic', 'Physical', 'Social/Civil']

    final_df = aiid_df[
        (aiid_df['mapped_sector'].isin(plot_sectors)) & 
        (aiid_df['derived_harm'].isin(plot_harms))
    ].copy()

    print(f"\nClassified Data Points: {len(final_df)}")
    print("Sector Counts:\n", final_df['mapped_sector'].value_counts())
    print("Harm Counts:\n", final_df['derived_harm'].value_counts())

    if len(final_df) > 10:
        # --- Statistics ---
        ct = pd.crosstab(final_df['mapped_sector'], final_df['derived_harm'])
        print("\n--- Contingency Table ---")
        print(ct)

        chi2, p, dof, expected = chi2_contingency(ct)
        print(f"\nChi-square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4e}")

        residuals = (ct - expected) / np.sqrt(expected)
        print("\n--- Standardized Residuals ---")
        print(residuals)

        # --- Visualization ---
        plt.figure(figsize=(10, 6))
        sns.heatmap(residuals, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
        plt.title(f"Standardized Residuals: Sector vs Derived Harm Type\n(Based on text analysis of '{text_col}')")
        plt.ylabel("Sector")
        plt.xlabel("Derived Harm Type")
        plt.tight_layout()
        plt.show()

        # --- Conclusion Logic ---
        print("\n--- Hypothesis Verification ---")
        try:
            fin_econ = residuals.loc['Financial', 'Economic']
            pub_soc = residuals.loc['Public Sector', 'Social/Civil']
            
            print(f"Financial -> Economic Residual: {fin_econ:.2f}")
            print(f"Public Sector -> Social/Civil Residual: {pub_soc:.2f}")
            
            if fin_econ > 1.96 and pub_soc > 1.96:
                print("Result: STRONG SUPPORT. Both sectors match the hypothesized harm profile significantly.")
            elif fin_econ > 1.96:
                print("Result: PARTIAL SUPPORT. Financial sector strongly linked to Economic harm.")
            elif pub_soc > 1.96:
                print("Result: PARTIAL SUPPORT. Public Sector strongly linked to Social/Civil harm.")
            else:
                print("Result: NO SIGNIFICANT SUPPORT. The hypothesized fingerprints were not strongly observed.")
        except KeyError as e:
            print(f"Could not verify hypothesis specific keys: {e}")
            
    else:
        print("Insufficient data after classification.")
