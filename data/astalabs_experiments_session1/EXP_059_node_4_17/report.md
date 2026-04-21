# Experiment 59: node_4_17

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_17` |
| **ID in Run** | 59 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:24:18.203186+00:00 |
| **Runtime** | 178.9s |
| **Parent** | `node_3_5` |
| **Children** | `node_5_15`, `node_5_22` |
| **Creation Index** | 60 |

---

## Hypothesis

> Governance requirements classified as 'Trust Readiness' map to a significantly
higher number of architecture controls per requirement than 'Integration
Readiness' requirements, indicating a higher degree of structural diffuseness.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.1963 (Likely False) |
| **Surprise** | -0.6333 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 30.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 3.0 |
| Definitely False | 87.0 |

---

## Experiment Plan

**Objective:** Compare the mean number of mapped architecture controls per governance requirement between the two bundles.

### Steps
- 1. Load 'step2_crosswalk_evidence.json'.
- 2. For each entry, calculate the length of the 'applicable_controls' list.
- 3. Group the data by the 'bundle' field.
- 4. Perform an Independent Samples T-test (Welch's t-test) to compare the mean control counts of 'Trust Readiness' vs. 'Integration Readiness'.

### Deliverables
- Mean control count per bundle, standard deviations, T-statistic, and p-value.

---

## Analysis

The experiment successfully tested the hypothesis that 'Trust Readiness'
governance requirements map to a significantly higher number of architecture
controls than 'Integration Readiness' requirements. Processing the
`step2_crosswalk_evidence.json` dataset (n=42), the analysis found that Trust
Readiness requirements mapped to a mean of 1.68 controls (SD=0.75), while
Integration Readiness requirements mapped to a mean of 2.00 controls (SD=0.52).
The Welch's t-test resulted in a t-statistic of -1.5519 and a p-value of 0.1308.
Since the p-value exceeds the alpha of 0.05, the difference is not statistically
significant. Consequently, the hypothesis is rejected; the data does not support
the claim that Trust requirements are more structurally diffuse. If anything,
the non-significant trend points in the opposite direction.

---

## Review

The experiment successfully tested the hypothesis that 'Trust Readiness'
governance requirements map to a significantly higher number of architecture
controls than 'Integration Readiness' requirements. Processing the
`step2_crosswalk_evidence.json` dataset (n=42), the analysis found that Trust
Readiness requirements mapped to a mean of 1.68 controls (SD=0.75), while
Integration Readiness requirements mapped to a mean of 2.00 controls (SD=0.52).
The Welch's t-test resulted in a t-statistic of -1.5519 and a p-value of 0.1308.
Since the p-value exceeds the alpha of 0.05, the difference is not statistically
significant. Consequently, the hypothesis is rejected; the data does not support
the claim that Trust requirements are more structurally diffuse. If anything,
the non-significant trend points in the opposite direction.

---

## Code

```python
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
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_crosswalk_evidence.json

--- Descriptive Statistics ---
Trust Readiness:       n=19, Mean=1.6842, Std Dev=0.7493
Integration Readiness: n=23, Mean=2.0000, Std Dev=0.5222

--- Welch's t-test Results ---
T-statistic: -1.5519
P-value: 0.1308

Result: No statistically significant difference detected (Hypothesis Rejected).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
