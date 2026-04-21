import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# [debug]
print("Starting experiment...")

# 1. Load the dataset
# Try loading from parent directory first, then current directory
file_name = 'astalabs_discovery_all_data.csv'
paths = [f'../{file_name}', file_name]
ds_path = None

for p in paths:
    if os.path.exists(p):
        ds_path = p
        break

if ds_path is None:
    print(f"Error: {file_name} not found in {paths}.")
    exit(1)

try:
    df = pd.read_csv(ds_path, low_memory=False)
    print(f"Dataset loaded successfully from {ds_path}.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# 2. Filter subsets
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()

print(f"AIID Incidents count: {len(aiid_df)}")
print(f"ATLAS Cases count: {len(atlas_df)}")

# 3. Normalization Logic
def normalize_sector(val):
    if pd.isna(val):
        return 'Other/Unknown'
    val_str = str(val).lower()
    
    # Check keywords mapping to standardized sectors
    if any(x in val_str for x in ['defense', 'military', 'government', 'public sector', 'security', 'police', 'surveillance']):
        return 'Defense/Govt'
    elif any(x in val_str for x in ['health', 'medical', 'hospital', 'biotech']):
        return 'Healthcare'
    elif any(x in val_str for x in ['finance', 'financial', 'bank', 'insurance', 'trading']):
        return 'Finance'
    elif any(x in val_str for x in ['consumer', 'retail', 'entertainment', 'media', 'social media', 'technology', 'internet', 'software', 'app']):
        return 'Consumer/Tech'
    elif any(x in val_str for x in ['transport', 'automotive', 'vehicle', 'aviation', 'driving', 'autonomous']):
        return 'Transportation'
    elif any(x in val_str for x in ['education', 'academic', 'school', 'university']):
        return 'Education'
    else:
        return 'Other/Unknown'

# Apply normalization
# AIID uses 'Sector of Deployment', ATLAS uses 'sector'
aiid_df['norm_sector'] = aiid_df['Sector of Deployment'].apply(normalize_sector)
atlas_df['norm_sector'] = atlas_df['sector'].apply(normalize_sector)

# 4. Calculate Distributions
aiid_counts = aiid_df['norm_sector'].value_counts()
atlas_counts = atlas_df['norm_sector'].value_counts()

# Align indices for comparison
all_sectors = sorted(list(set(aiid_counts.index) | set(atlas_counts.index)))

# Create a DataFrame for the contingency table (Counts)
comparison_df = pd.DataFrame(index=all_sectors)
comparison_df['AIID'] = comparison_df.index.map(aiid_counts).fillna(0).astype(int)
comparison_df['ATLAS'] = comparison_df.index.map(atlas_counts).fillna(0).astype(int)

print("\nSector Distribution (Counts):")
print(comparison_df)

# Calculate Percentages for plotting
plot_df = comparison_df.copy()
plot_df['AIID_pct'] = plot_df['AIID'] / plot_df['AIID'].sum() * 100
plot_df['ATLAS_pct'] = plot_df['ATLAS'] / plot_df['ATLAS'].sum() * 100

print("\nSector Distribution (Percentages):")
print(plot_df[['AIID_pct', 'ATLAS_pct']])

# 5. Statistical Test: Chi-Square Test of Homogeneity
# We check if the distribution of sectors depends on the dataset source.
# Transpose so rows are [AIID, ATLAS] and columns are sectors
chi2, p, dof, expected = chi2_contingency(comparison_df[['AIID', 'ATLAS']].T)

print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"p-value: {p:.4e}")
print(f"Degrees of Freedom: {dof}")

interpretation = "Significantly different" if p < 0.05 else "Not significantly different"
print(f"Conclusion: The sector distributions are {interpretation}.")

# 6. Visualization
fig, ax = plt.subplots(figsize=(12, 6))

width = 0.35
x_indices = np.arange(len(all_sectors))

# Plot bars
rects1 = ax.bar(x_indices - width/2, plot_df['AIID_pct'], width, label='AIID (Real Incidents)', alpha=0.8)
rects2 = ax.bar(x_indices + width/2, plot_df['ATLAS_pct'], width, label='ATLAS (Adversarial Research)', alpha=0.8)

ax.set_ylabel('Percentage of Cases')
ax.set_title('Sector Distribution: Real-World Incidents vs. Adversarial Research')
ax.set_xticks(x_indices)
ax.set_xticklabels(all_sectors, rotation=45, ha='right')
ax.legend()

# Add text labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()