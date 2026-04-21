import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load Data
try:
    df = pd.read_csv('../step2_competency_statements.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset not found at '../step2_competency_statements.csv'. Checking current directory...")
    try:
        df = pd.read_csv('step2_competency_statements.csv')
        print("Dataset loaded from current directory.")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# 2. Preprocessing
# Map confidence to numeric values
# Assuming values might be 'High', 'Medium', 'Low' (case-insensitive)
confidence_map = {'high': 3, 'medium': 2, 'low': 1}
df['confidence_cleaned'] = df['confidence'].astype(str).str.strip().str.lower()
df['confidence_score'] = df['confidence_cleaned'].map(confidence_map)

# Debug: Check for unmapped values
unmapped = df[df['confidence_score'].isna()]
if not unmapped.empty:
    print(f"Warning: {len(unmapped)} records have unmapped confidence values:")
    print(unmapped['confidence'].unique())
    # Drop them for analysis
    df = df.dropna(subset=['confidence_score'])

# 3. Grouping
trust_scores = df[df['bundle'] == 'Trust Readiness']['confidence_score']
integration_scores = df[df['bundle'] == 'Integration Readiness']['confidence_score']

# 4. Descriptive Statistics
mean_trust = trust_scores.mean()
std_trust = trust_scores.std(ddof=1)
median_trust = trust_scores.median()
n_trust = len(trust_scores)

mean_integration = integration_scores.mean()
std_integration = integration_scores.std(ddof=1)
median_integration = integration_scores.median()
n_integration = len(integration_scores)

print("\n--- Descriptive Statistics ---")
print(f"Trust Readiness (n={n_trust}): Mean={mean_trust:.2f}, Median={median_trust}, Std={std_trust:.2f}")
print(f"Integration Readiness (n={n_integration}): Mean={mean_integration:.2f}, Median={median_integration}, Std={std_integration:.2f}")

# Cross-tabulation for detailed view
print("\n--- Confidence Level Distribution ---")
ct = pd.crosstab(df['bundle'], df['confidence_cleaned'])
# Reorder columns if they exist
order = [col for col in ['low', 'medium', 'high'] if col in ct.columns]
print(ct[order])

# 5. Statistical Test (Mann-Whitney U)
# Using Mann-Whitney U because data is ordinal and likely not normally distributed
u_stat, p_val = stats.mannwhitneyu(trust_scores, integration_scores, alternative='two-sided')

print("\n--- Mann-Whitney U Test Results ---")
print(f"U-statistic: {u_stat}")
print(f"P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Conclusion: Reject the null hypothesis. There is a statistically significant difference in confidence levels.")
else:
    print("Conclusion: Fail to reject the null hypothesis. No statistically significant difference in confidence levels detected.")

# 6. Visualization
labels = ['Trust Readiness', 'Integration Readiness']
means = [mean_trust, mean_integration]
stds = [std_trust, std_integration]

plt.figure(figsize=(8, 6))
# Use standard error for error bars (std / sqrt(n))
se_trust = std_trust / np.sqrt(n_trust)
se_integration = std_integration / np.sqrt(n_integration)
errors = [se_trust, se_integration]

bars = plt.bar(labels, means, yerr=errors, capsize=10, color=['#4c72b0', '#55a868'], alpha=0.8)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.title('Mean Evidence Confidence Score by Competency Bundle')
plt.ylabel('Confidence Score (1=Low, 2=Med, 3=High)')
plt.ylim(0, 3.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
