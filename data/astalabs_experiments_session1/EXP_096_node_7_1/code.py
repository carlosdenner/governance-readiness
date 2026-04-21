import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re

# Define the file path based on the instruction that datasets are one level above
file_path = '../step3_incident_coding.csv'

# Fallback to current directory if the file is not found in the parent directory
if not os.path.exists(file_path):
    if os.path.exists('step3_incident_coding.csv'):
        file_path = 'step3_incident_coding.csv'
    else:
        print("Warning: step3_incident_coding.csv not found in ../ or ./")

# Load the dataset
print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Function to extract and count unique Trust (TR) and Integration (IR) competency gaps
def count_competencies(val):
    if pd.isna(val):
        return 0, 0
    val_str = str(val)
    # Regex to find codes like TR-1, TR-01, IR-5, IR-08, etc.
    # Using set() to ensure we count unique competencies per incident
    tr_matches = set(re.findall(r'TR-\d+', val_str, re.IGNORECASE))
    ir_matches = set(re.findall(r'IR-\d+', val_str, re.IGNORECASE))
    return len(tr_matches), len(ir_matches)

# Apply the counting function
df[['Trust_Count', 'Integration_Count']] = df['llm_sub_competencies'].apply(
    lambda x: pd.Series(count_competencies(x))
)

# Perform Correlation Analysis
pearson_r, pearson_p = stats.pearsonr(df['Trust_Count'], df['Integration_Count'])
spearman_r, spearman_p = stats.spearmanr(df['Trust_Count'], df['Integration_Count'])

# Print Summary Statistics
print("=== Summary Statistics ===")
print(df[['Trust_Count', 'Integration_Count']].describe())
print("\n=== Correlation Analysis ===")
print(f"Pearson Correlation (r): {pearson_r:.4f} (p-value: {pearson_p:.4f})")
print(f"Spearman Correlation (rho): {spearman_r:.4f} (p-value: {spearman_p:.4f})")

# Visualization: Scatter Plot with Regression Line
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Use jitter to prevent overplotting of integer values
ax = sns.regplot(
    data=df,
    x='Trust_Count',
    y='Integration_Count',
    x_jitter=0.2,
    y_jitter=0.2,
    scatter_kws={'alpha': 0.6, 's': 60, 'edgecolor': 'white'},
    line_kws={'color': 'red', 'label': f'Linear Fit (r={pearson_r:.2f})'}
)

plt.title('Correlation: Trust vs. Integration Competency Deficiencies', fontsize=14)
plt.xlabel('Number of Trust Readiness Gaps (TR)', fontsize=12)
plt.ylabel('Number of Integration Readiness Gaps (IR)', fontsize=12)
plt.legend()

# Annotate with statistical results
stats_text = f'Pearson r = {pearson_r:.2f} (p={pearson_p:.3f})\nSpearman rho = {spearman_r:.2f} (p={spearman_p:.3f})'
plt.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()