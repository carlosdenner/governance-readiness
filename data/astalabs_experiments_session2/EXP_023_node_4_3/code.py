import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os
import re

# [debug]
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))
try:
    print("Files in parent directory:", os.listdir('..'))
except Exception as e:
    print("Cannot list parent directory:", e)

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
filepath = f'../{filename}' if os.path.exists(f'../{filename}') else filename

try:
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Successfully loaded {filepath}")
except FileNotFoundError:
    print(f"Error: Could not find {filename} in . or ..")
    exit(1)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid)}")

# Clean Autonomy Level
# Expecting values that can be mapped to ordinal. 
# Previous context suggests 'Autonomy1', 'Autonomy2', etc.
print("\nUnique Autonomy Levels (raw):", aiid['Autonomy Level'].unique())

def parse_autonomy(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    # Look for digits
    digits = re.findall(r'\d+', val_str)
    if digits:
        return int(digits[0])
    # Fallback keyword mapping if needed (though datasets usually use numeric codes)
    if 'high' in val_str: return 3
    if 'medium' in val_str: return 2
    if 'low' in val_str: return 1
    return None

aiid['Autonomy_Ordinal'] = aiid['Autonomy Level'].apply(parse_autonomy)

# Clean Technical Failure
print("\nUnique Technical Failures (raw top 20):", aiid['Known AI Technical Failure'].value_counts().head(20).index.tolist())

def classify_failure(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    
    # Keywords for Robustness
    robust_keys = ['robust', 'reliability', 'dependability', 'sensor', 'noise', 'environment', 'shift', 'distribution', 'failure of mechanism']
    # Keywords for Operator/HMI
    operator_keys = ['human', 'operator', 'user', 'mistake', 'misuse', 'hmi', 'interaction', 'training']
    
    is_robust = any(k in val_str for k in robust_keys)
    is_operator = any(k in val_str for k in operator_keys)
    
    if is_robust and not is_operator:
        return 'Robustness'
    elif is_operator and not is_robust:
        return 'Operator/HMI'
    elif is_robust and is_operator:
        return 'Mixed'
    else:
        return 'Other'

aiid['Failure_Category'] = aiid['Known AI Technical Failure'].apply(classify_failure)

# Filter data for analysis
analysis_df = aiid.dropna(subset=['Autonomy_Ordinal', 'Failure_Category'])
analysis_df = analysis_df[analysis_df['Failure_Category'].isin(['Robustness', 'Operator/HMI'])]

print("\nData for Analysis:")
print(analysis_df['Failure_Category'].value_counts())
print(analysis_df.groupby('Failure_Category')['Autonomy_Ordinal'].describe())

# Statistical Test
robustness_scores = analysis_df[analysis_df['Failure_Category'] == 'Robustness']['Autonomy_Ordinal']
operator_scores = analysis_df[analysis_df['Failure_Category'] == 'Operator/HMI']['Autonomy_Ordinal']

if len(robustness_scores) > 0 and len(operator_scores) > 0:
    u_stat, p_val = stats.mannwhitneyu(robustness_scores, operator_scores, alternative='two-sided')
    print(f"\nMann-Whitney U Test:\nU-statistic: {u_stat}\nP-value: {p_val}")
else:
    print("\nInsufficient data for statistical test.")

# Visualization
plt.figure(figsize=(10, 6))
sns.violinplot(data=analysis_df, x='Failure_Category', y='Autonomy_Ordinal', inner='stick', palette='muted')
plt.title('Distribution of Autonomy Levels by Failure Category')
plt.ylabel('Autonomy Level (Ordinal)')
plt.xlabel('Failure Category')
plt.grid(axis='y', alpha=0.3)
plt.show()