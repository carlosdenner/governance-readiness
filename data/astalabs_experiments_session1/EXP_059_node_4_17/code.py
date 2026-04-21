import json
import os
import numpy as np
from scipy import stats

file_name = 'step2_crosswalk_evidence.json'
file_path = file_name

# Check if file exists in current directory, otherwise check parent directory
if not os.path.exists(file_path):
    if os.path.exists(os.path.join('..', file_name)):
        file_path = os.path.join('..', file_name)

print(f"Loading dataset from: {file_path}")

try:
    with open(file_path, 'r') as f:
        data = json.load(f)

    trust_counts = []
    integration_counts = []

    for entry in data:
        bundle = entry.get('bundle')
        controls = entry.get('applicable_controls', [])
        
        # Determine count based on type (list or semicolon-separated string)
        if isinstance(controls, list):
            count = len(controls)
        elif isinstance(controls, str):
            count = len([c for c in controls.split(';') if c.strip()])
        else:
            count = 0
            
        if bundle == 'Trust Readiness':
            trust_counts.append(count)
        elif bundle == 'Integration Readiness':
            integration_counts.append(count)

    # Convert to numpy arrays
    trust_arr = np.array(trust_counts)
    integration_arr = np.array(integration_counts)

    # Calculate descriptive statistics
    trust_n = len(trust_arr)
    trust_mean = np.mean(trust_arr) if trust_n > 0 else 0
    trust_std = np.std(trust_arr, ddof=1) if trust_n > 1 else 0
    
    integration_n = len(integration_arr)
    integration_mean = np.mean(integration_arr) if integration_n > 0 else 0
    integration_std = np.std(integration_arr, ddof=1) if integration_n > 1 else 0

    print(f"\n--- Descriptive Statistics ---")
    print(f"Trust Readiness:       n={trust_n}, Mean={trust_mean:.4f}, Std Dev={trust_std:.4f}")
    print(f"Integration Readiness: n={integration_n}, Mean={integration_mean:.4f}, Std Dev={integration_std:.4f}")

    # Perform Welch's t-test (equal_var=False for unequal variances/sample sizes)
    t_stat, p_val = stats.ttest_ind(trust_arr, integration_arr, equal_var=False)

    print(f"\n--- Welch's t-test Results ---")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4f}")

    # Interpret results regarding the hypothesis
    # Hypothesis: Trust > Integration (Significant)
    alpha = 0.05
    if p_val < alpha:
        print("\nResult: Statistically significant difference detected.")
        if t_stat > 0:
            print("Direction: Trust Readiness maps to MORE controls (Hypothesis Supported).")
        else:
            print("Direction: Trust Readiness maps to FEWER controls (Hypothesis Rejected).")
    else:
        print("\nResult: No statistically significant difference detected (Hypothesis Rejected).")

except Exception as e:
    print(f"An error occurred: {e}")