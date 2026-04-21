import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import numpy as np

# Define file names
file_step1 = 'step1_sub_competencies.csv'
file_step3 = 'step3_coverage_map.csv'

# Resolve paths (check current and parent directory)
def get_path(filename):
    if os.path.exists(filename):
        return filename
    elif os.path.exists(os.path.join('..', filename)):
        return os.path.join('..', filename)
    return filename

path_step1 = get_path(file_step1)
path_step3 = get_path(file_step3)

# Load datasets
try:
    df_definitions = pd.read_csv(path_step1)
    df_coverage = pd.read_csv(path_step3)
    print(f"Loaded {file_step1} with shape {df_definitions.shape}")
    print(f"Loaded {file_step3} with shape {df_coverage.shape}")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# Calculate semantic breadth (word count) for definitions
# Ensure definition column is string
df_definitions['definition'] = df_definitions['definition'].astype(str)
df_definitions['word_count'] = df_definitions['definition'].apply(lambda x: len(x.split()))

# Merge datasets on ID
# step1 uses 'id', step3 uses 'sub_competency_id'
merged_df = pd.merge(
    df_definitions[['id', 'name', 'word_count']],
    df_coverage[['sub_competency_id', 'incident_count', 'coverage_status']],
    left_on='id',
    right_on='sub_competency_id',
    how='inner'
)

print(f"\nMerged dataset shape: {merged_df.shape}")
if merged_df.empty:
    print("No overlapping IDs found between datasets.")
    exit(1)

print("\n--- Sample Data ---")
print(merged_df[['id', 'word_count', 'incident_count']].head())

# Statistical Correlation
x = merged_df['word_count']
y = merged_df['incident_count']

pearson_r, pearson_p = stats.pearsonr(x, y)
spearman_r, spearman_p = stats.spearmanr(x, y)

print("\n--- Statistical Results ---")
print(f"Pearson Correlation: r={pearson_r:.3f}, p={pearson_p:.3f}")
print(f"Spearman Correlation: r={spearman_r:.3f}, p={spearman_p:.3f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Sub-competencies')

# Annotate points
for i, txt in enumerate(merged_df['id']):
    plt.annotate(txt, (x.iloc[i], y.iloc[i]), xytext=(5, 5), textcoords='offset points')

# Regression line
if len(merged_df) > 1:
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--', label=f'Fit: y={m:.2f}x + {b:.2f}')

plt.title('Definition Semantic Breadth vs. Incident Coverage')
plt.xlabel('Definition Word Count')
plt.ylabel('Incident Count')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()