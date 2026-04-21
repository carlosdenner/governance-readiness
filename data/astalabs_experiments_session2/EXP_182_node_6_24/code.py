import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# --- Step 1: Load Dataset ---
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(f'../{filename}'):
    filepath = f'../{filename}'
else:
    filepath = filename # Fail gracefully (or with error) if not found

print(f"Loading dataset from: {filepath}")
df = pd.read_csv(filepath, low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 rows: {len(eo_data)}")

# --- Step 2: Parse Dates ---
def parse_year(date_str):
    if pd.isna(date_str):
        return np.nan
    try:
        dt = pd.to_datetime(date_str, errors='coerce')
        if pd.notnull(dt):
            return dt.year
    except:
        pass
    return np.nan

eo_data['initiation_year'] = eo_data['18_date_initiated'].apply(parse_year)
eo_data_clean = eo_data.dropna(subset=['initiation_year']).copy()
print(f"Rows with valid initiation year: {len(eo_data_clean)}")

# --- Step 3: Classify Disparity Mitigation (Text Analysis) ---
# Heuristic: If text describes a process, it's a 1. If it says 'N/A', 'None', or is empty, it's a 0.

def classify_mitigation(val):
    if pd.isna(val):
        return 0
    
    text = str(val).strip().lower()
    
    # Check for empty or very short strings
    if len(text) < 4:
        return 0
        
    # Check for explicit negatives at the start
    negative_prefixes = ('n/a', 'na ', 'no ', 'none', 'not applicable', 'unknown', 'tbd')
    if text.startswith(negative_prefixes):
        return 0
        
    # Also check if the entire string is just 'no' or 'none' (handled by starts with logic mostly, but 'no' < 4 chars handled above)
    
    # If we are here, it likely contains descriptive text of a mitigation
    return 1

eo_data_clean['mitigation_score'] = eo_data_clean['62_disparity_mitigation'].apply(classify_mitigation)

# Debug: Check distribution
print("\nMitigation Score Distribution:")
print(eo_data_clean['mitigation_score'].value_counts())

# Debug: Show examples of 1s and 0s to verify heuristic
print("\nExamples of '1' (Mitigation Present):")
print(eo_data_clean[eo_data_clean['mitigation_score']==1]['62_disparity_mitigation'].head(3).tolist())
print("\nExamples of '0' (No Mitigation/NA):")
print(eo_data_clean[eo_data_clean['mitigation_score']==0]['62_disparity_mitigation'].head(3).tolist())

# --- Step 4: Split Groups & Statistical Test ---
group_pre = eo_data_clean[eo_data_clean['initiation_year'] <= 2020]
group_post = eo_data_clean[eo_data_clean['initiation_year'] >= 2021]

score_pre = group_pre['mitigation_score']
score_post = group_post['mitigation_score']

print(f"\nGroup Pre-2020 (<= 2020): N={len(score_pre)}")
print(f"Group Post-2020 (>= 2021): N={len(score_post)}")

mean_pre = score_pre.mean()
mean_post = score_post.mean()

print(f"Mean Compliance Pre-2020: {mean_pre:.4f}")
print(f"Mean Compliance Post-2020: {mean_post:.4f}")

# T-test
t_stat, p_val = stats.ttest_ind(score_post, score_pre, equal_var=False, alternative='greater')
print(f"\nT-test results (Post > Pre):")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# --- Step 5: Visualization ---
plt.figure(figsize=(10, 6))
means = [mean_pre, mean_post]
labels = ['Pre-2020 (<=2020)', 'Post-2020 (>=2021)']

# Calculate Standard Error
se_pre = score_pre.sem()
se_post = score_post.sem()

bars = plt.bar(labels, means, yerr=[se_pre, se_post], capsize=10, 
               color=['#cccccc', '#2ca02c'], alpha=0.9, width=0.6)

plt.ylabel('Proportion with Disparity Mitigation Description')
plt.title('The "Awakening" Lag: Federal AI Bias Mitigation (Pre vs Post 2020)')
plt.ylim(0, 1.0)

# Add annotations
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, 
             f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
