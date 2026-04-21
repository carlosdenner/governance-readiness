import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define the file path (handling the directory note)
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    filepath = filename

print(f"Loading dataset from {filepath}...")

# Load dataset
df = pd.read_csv(filepath, low_memory=False)

# Filter for EO 13960 scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Define Agency Categories
science_agencies = ['NASA', 'DOE', 'NSF', 'HHS']
security_agencies = ['DHS', 'DOD', 'DOJ', 'STATE']

def categorize_agency(abr):
    if abr in science_agencies:
        return 'Science'
    elif abr in security_agencies:
        return 'Security'
    else:
        return 'Other'

df_eo['Agency_Category'] = df_eo['3_abr'].apply(categorize_agency)

# Filter for only the two target categories
df_target = df_eo[df_eo['Agency_Category'] != 'Other'].copy()

print(f"Filtered Dataset Shape: {df_target.shape}")
print(f"Counts per category:\n{df_target['Agency_Category'].value_counts()}")

# Define governance columns
gov_columns = [
    '52_impact_assessment',
    '53_real_world_testing',
    '56_monitor_postdeploy',
    '62_disparity_mitigation',
    '59_ai_notice'
]

# Helper function to binarize governance responses
def is_affirmative(val, col_name=None):
    if pd.isna(val):
        return 0
    text = str(val).lower().strip()
    
    # Negative keywords
    negatives = ['no', 'none', 'n/a', 'not', 'waived', 'pending', 'unknown']
    
    # Specific logic for Notice (based on previous exploration)
    if col_name == '59_ai_notice':
        # If it starts with a negative indicator
        if any(text.startswith(n) for n in negatives):
            return 0
        # If it contains "none of the above"
        if "none of the above" in text:
            return 0
        return 1
    
    # General logic for other columns (Assessment, Testing, etc.)
    # Usually look for "yes", "completed", "conducted"
    # If strict "no" or "not applicable", then 0.
    
    # Check for explicit No first
    if text in ['no', 'no.', 'not applicable', 'n/a', 'none']:
        return 0
    
    # Check for Yes/Positive indicators
    positives = ['yes', 'completed', 'conducted', 'performed', 'implemented', 'ongoing']
    if any(p in text for p in positives):
        return 1
        
    # Fallback: if not explicitly negative, assume 0 for safety unless unclear, 
    # but let's see. Many fields might be descriptive.
    # For this experiment, we'll assume if it doesn't match positives, it's 0.
    return 0

# Calculate scores
for col in gov_columns:
    df_target[f'Score_{col}'] = df_target[col].apply(lambda x: is_affirmative(x, col))

# Sum for composite score
df_target['Governance_Score'] = df_target[[f'Score_{c}' for c in gov_columns]].sum(axis=1)

# Separate groups
science_scores = df_target[df_target['Agency_Category'] == 'Science']['Governance_Score']
security_scores = df_target[df_target['Agency_Category'] == 'Security']['Governance_Score']

# Statistical Test (Mann-Whitney U)
u_stat, p_val = stats.mannwhitneyu(science_scores, security_scores, alternative='two-sided')

# Calculate Means
mean_science = science_scores.mean()
mean_security = security_scores.mean()

print("\n--- Results ---")
print(f"Science Agencies (n={len(science_scores)}) Mean Governance Score: {mean_science:.2f}")
print(f"Security Agencies (n={len(security_scores)}) Mean Governance Score: {mean_security:.2f}")
print(f"Mann-Whitney U Statistic: {u_stat}, p-value: {p_val:.5f}")

if p_val < 0.05:
    print("Result: Statistically significant difference.")
else:
    print("Result: No statistically significant difference.")

# Visualization
plt.figure(figsize=(10, 6))
# Create a list of data for boxplot
data_to_plot = [science_scores, security_scores]
labels = ['Science (NASA, DOE, NSF, HHS)', 'Security (DHS, DOD, DOJ, STATE)']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
plt.title('Governance Readiness Scores by Agency Type')
plt.ylabel('Composite Governance Score (0-5)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text annotation for p-value
plt.text(1.5, 4.5, f'p={p_val:.4f}', horizontalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.show()
