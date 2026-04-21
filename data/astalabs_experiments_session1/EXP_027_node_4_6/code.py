import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np
import os

# Robust file loading strategy
filename = 'step2_competency_statements.csv'
file_path = filename # Default to current directory

if not os.path.exists(file_path):
    # Check parent directory if not found in current
    if os.path.exists(f'../{filename}'):
        file_path = f'../{filename}'
    else:
        # Fallback to current which will raise error, or print warning
        print(f"Warning: {filename} not found in current or parent directory. Attempting current.")

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# 1. Derive control_density
# The applicable_controls column contains semicolon-separated strings.
def count_controls(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    return len([x for x in str(val).split(';') if x.strip()])

df['control_density'] = df['applicable_controls'].apply(count_controls)

# 2. Map confidence to numeric
# Normalize text to title case to handle potential inconsistencies (e.g. 'high', 'High')
confidence_map = {'High': 3, 'Medium': 2, 'Low': 1}
df['confidence_norm'] = df['confidence'].astype(str).str.strip().str.title()
df['confidence_numeric'] = df['confidence_norm'].map(confidence_map)

# Filter out any rows where confidence couldn't be mapped (if any)
df_clean = df.dropna(subset=['confidence_numeric'])

print(f"Analyzable records: {len(df_clean)}")

# 3. Calculate Spearman Correlation
# We use Spearman because 'confidence' is ordinal
corr, p_value = spearmanr(df_clean['control_density'], df_clean['confidence_numeric'])

print(f"\nSpearman Correlation Coefficient: {corr:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant correlation.")
else:
    print("Result: No statistically significant correlation.")

# 4. Visualization
plt.figure(figsize=(10, 6))

# Add jitter for visualization purposes since data is discrete
jitter_x = np.random.uniform(-0.15, 0.15, size=len(df_clean))
jitter_y = np.random.uniform(-0.15, 0.15, size=len(df_clean))

plt.scatter(
    df_clean['control_density'] + jitter_x, 
    df_clean['confidence_numeric'] + jitter_y, 
    alpha=0.6, 
    c='teal', 
    s=100, 
    edgecolors='black'
)

plt.title(f'Control Density vs. Evidence Confidence\n(Spearman r={corr:.2f}, p={p_value:.3f})')
plt.xlabel('Control Density (Number of Architecture Controls)')
plt.ylabel('Evidence Confidence')
plt.yticks([1, 2, 3], ['Low (1)', 'Medium (2)', 'High (3)'])
plt.grid(True, linestyle='--', alpha=0.5)

# Add a trend line (linear regression fit) just for visual aid, even if correlation is non-parametric
sns.regplot(
    x=df_clean['control_density'], 
    y=df_clean['confidence_numeric'], 
    scatter=False, 
    color='darkred', 
    line_kws={'linestyle': ':'}
)

plt.tight_layout()
plt.show()

# Print detailed frequency table to help explain the result
print("\nContingency Table (Counts):")
crosstab = pd.crosstab(df_clean['confidence_norm'], df_clean['control_density'])
print(crosstab)