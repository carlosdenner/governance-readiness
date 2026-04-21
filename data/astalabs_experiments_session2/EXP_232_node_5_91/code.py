import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

# [debug] Print current directory and list files to ensure path is correct
# print(f"Current working directory: {os.getcwd()}")
# print(f"Files in parent directory: {os.listdir('..')}")

# Load dataset
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'  # Fallback for local testing

try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID incidents loaded: {len(aiid)}")

# Define column names based on metadata
col_failure = 'Known AI Technical Failure'
col_sector = 'Sector of Deployment'

# Check for column existence
missing_cols = [c for c in [col_failure, col_sector] if c not in aiid.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}. Available columns sample: {aiid.columns[:10]}")
    sys.exit(1)

# --- Step 1: Clean and Map Failure Types ---
# Inspect unique values to guide mapping (printing top 20)
print("\n--- Top 20 Raw Failure Types ---")
print(aiid[col_failure].value_counts(dropna=False).head(20))

def map_failure_type(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    
    # Mapping based on hypothesis definitions
    if 'specification' in val_str:
        return 'Specification'
    elif 'robustness' in val_str or 'adversarial' in val_str:
        return 'Robustness'
    # Some definitions map 'reliability' or 'error' to robustness in broad terms, 
    # but we stick to strict keywords first.
    return 'Other'

aiid['Failure_Class'] = aiid[col_failure].apply(map_failure_type)

# --- Step 2: Clean and Map Sectors ---
# Inspect unique values
print("\n--- Top 20 Raw Sectors ---")
print(aiid[col_sector].value_counts(dropna=False).head(20))

def map_sector_group(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).lower().strip()
    
    # Hypothesis Group 1: Social Media & Advertising
    if any(k in val_str for k in ['social media', 'advertising', 'entertainment', 'news', 'media']):
        return 'Social/Media/Ad'
    
    # Hypothesis Group 2: Security & Industrial
    if any(k in val_str for k in ['security', 'defense', 'industrial', 'manufacturing', 'robotics', 'military', 'surveillance']):
        return 'Security/Industrial'
    
    # Other distinct groups for context
    if any(k in val_str for k in ['transportation', 'automotive', 'vehicle']):
        return 'Transportation'
    if any(k in val_str for k in ['healthcare', 'medicine', 'hospital']):
        return 'Healthcare'
    if any(k in val_str for k in ['financial', 'finance', 'banking']):
        return 'Finance'
        
    return 'Other'

aiid['Sector_Class'] = aiid[col_sector].apply(map_sector_group)

# --- Step 3: Filter Data for Analysis ---
# We focus on rows that have a valid mapped Failure Class (Specification or Robustness)
df_analysis = aiid[aiid['Failure_Class'].isin(['Specification', 'Robustness'])].copy()

# We remove 'Unknown' sectors to clean up the plot, but keep 'Other' for comparison
df_analysis = df_analysis[df_analysis['Sector_Class'] != 'Unknown']

print(f"\nData points remaining for analysis (Specification vs Robustness): {len(df_analysis)}")

if len(df_analysis) < 5:
    print("Insufficient data for statistical analysis.")
    sys.exit(0)

# --- Step 4: Statistical Analysis (Chi-Square) ---
contingency_table = pd.crosstab(df_analysis['Sector_Class'], df_analysis['Failure_Class'])
print("\n--- Contingency Table (Sector vs Failure Type) ---")
print(contingency_table)

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Degrees of Freedom: {dof}")

# Calculate Standardized Residuals to identify drivers of significance
# Residual = (Observed - Expected) / sqrt(Expected)
std_residuals = (contingency_table - expected) / np.sqrt(expected)
print("\n--- Standardized Residuals (Values > 1.96 or < -1.96 differ significantly) ---")
print(std_residuals)

# --- Step 5: Visualization ---
# Plot proportions
props = contingency_table.div(contingency_table.sum(axis=1), axis=0)

ax = props.plot(kind='barh', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title('Proportion of Failure Types by Sector')
plt.xlabel('Proportion')
plt.ylabel('Sector Group')
plt.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
