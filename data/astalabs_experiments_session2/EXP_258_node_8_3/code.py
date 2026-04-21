import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import sys
import os

print("Starting Adversarial Governance Gap experiment (Attempt 3)...")

# 1. Load Dataset
dataset_path = 'astalabs_discovery_all_data.csv'
if not os.path.exists(dataset_path):
    # Check parent directory just in case
    dataset_path = '../astalabs_discovery_all_data.csv'
    if not os.path.exists(dataset_path):
        print("Error: Dataset not found.")
        sys.exit(1)

try:
    df = pd.read_csv(dataset_path, low_memory=False)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# 2. Filter for Incident Coding
subset = df[df['source_table'] == 'step3_incident_coding'].copy()
print(f"Subset shape: {subset.shape}")

# 3. Parse 'competency_domains' (identified in debug as the correct column)
# Drop NaNs first
subset = subset.dropna(subset=['competency_domains'])
print(f"Rows with valid competency_domains: {len(subset)}")

tr_count = 0
ir_count = 0

for entry in subset['competency_domains']:
    # Split by semicolon as seen in debug output
    domains = [d.strip() for d in str(entry).split(';')]
    
    for domain in domains:
        # Check for keywords "Trust Readiness" vs "Integration Readiness"
        if "Trust Readiness" in domain:
            tr_count += 1
        elif "Integration Readiness" in domain:
            ir_count += 1

print(f"\nCounts:\nTrust Readiness (TR): {tr_count}\nIntegration Readiness (IR): {ir_count}")

# 4. Statistical Test (Chi-Square Goodness of Fit)
# Null Hypothesis: TR and IR gaps appear with equal frequency
total_observations = tr_count + ir_count

if total_observations > 0:
    expected = [total_observations / 2, total_observations / 2]
    observed = [tr_count, ir_count]
    
    chi2_stat, p_val = chisquare(f_obs=observed, f_exp=expected)
    
    print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_val:.4e}")
    
    alpha = 0.05
    if p_val < alpha:
        print("Result: Statistically Significant deviation from equal frequency.")
        if tr_count > ir_count:
            print("Direction: Trust Readiness gaps are significantly more prevalent.")
        else:
            print("Direction: Integration Readiness gaps are significantly more prevalent.")
    else:
        print("Result: No statistically significant difference between TR and IR prevalence.")

    # 5. Visualization
    labels = ['Trust Readiness', 'Integration Readiness']
    counts = [tr_count, ir_count]
    colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=colors)
    
    plt.ylabel('Frequency of Competency Gaps')
    plt.title('Adversarial Governance Gap: Trust vs. Integration Failures')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')
    
    plt.show()
else:
    print("No relevant domains found to analyze.")
