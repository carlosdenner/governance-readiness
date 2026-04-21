import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Define file path
file_name = 'step3_enrichments.json'
file_path = f'../{file_name}' if os.path.exists(f'../{file_name}') else file_name

# Load dataset
print(f"Loading dataset from: {file_path}")
try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File {file_path} not found.")
    exit(1)

# Process data
records = []
for entry in data:
    sub_competencies = entry.get('sub_competency_ids', [])
    technique_count = entry.get('technique_count', 0)
    
    # Handle cases where sub_competency_ids might be a string (CSV parsing artifact) or list
    if isinstance(sub_competencies, str):
        sub_competencies = [x.strip() for x in sub_competencies.split(';') if x.strip()]
    
    if not isinstance(sub_competencies, list):
        sub_competencies = []
        
    ir_count = sum(1 for cid in sub_competencies if cid.startswith('IR-'))
    total_count = len(sub_competencies)
    
    if total_count > 0:
        integration_ratio = ir_count / total_count
        records.append({
            'case_study_id': entry.get('case_study_id'),
            'technique_count': technique_count,
            'integration_ratio': integration_ratio,
            'sub_competency_count': total_count
        })

df = pd.DataFrame(records)

print(f"Processed {len(df)} incidents with valid competency mappings.")

# Analysis 1: Correlation
pearson_corr, p_value_corr = stats.pearsonr(df['integration_ratio'], df['technique_count'])
print(f"\nCorrelation (Integration Ratio vs Technique Count):")
print(f"  Pearson r: {pearson_corr:.4f}")
print(f"  P-value: {p_value_corr:.4f}")

# Analysis 2: Group Comparison
# Split into High Integration (> 0.5) and High Trust (<= 0.5)
df['group'] = df['integration_ratio'].apply(lambda x: 'High Integration' if x > 0.5 else 'High Trust')

group_counts = df['group'].value_counts()
print(f"\nGroup Sizes:\n{group_counts}")

high_integration_scores = df[df['group'] == 'High Integration']['technique_count']
high_trust_scores = df[df['group'] == 'High Trust']['technique_count']

# T-test
t_stat, p_value_ttest = stats.ttest_ind(high_integration_scores, high_trust_scores, equal_var=False)

print(f"\nGroup Statistics (Technique Count):")
print(f"  High Integration Mean: {high_integration_scores.mean():.2f} (std: {high_integration_scores.std():.2f})")
print(f"  High Trust Mean:       {high_trust_scores.mean():.2f} (std: {high_trust_scores.std():.2f})")
print(f"\nT-test results (Welch's):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_ttest:.4f}")

# Visualization
plt.figure(figsize=(10, 6))

# Boxplot
boxplot_data = [high_trust_scores, high_integration_scores]
plt.boxplot(boxplot_data, labels=['High Trust (Ratio <= 0.5)', 'High Integration (Ratio > 0.5)'])
plt.title('Attack Complexity (Technique Count) by Competency Domain')
plt.ylabel('Technique Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add scatter points for visibility
for i, d in enumerate(boxplot_data, start=1):
    y = d
    x = np.random.normal(i, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.5)

plt.show()

# Interpretation
alpha = 0.05
print("\n=== Conclusion ===")
if p_value_ttest < alpha:
    if t_stat > 0:
        print("Result: Statistically significant. Integration-focused incidents involve MORE techniques.")
    else:
        print("Result: Statistically significant. Integration-focused incidents involve FEWER techniques.")
else:
    print("Result: No statistically significant difference in technique count between groups.")
