# Experiment 258: node_8_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_8_3` |
| **ID in Run** | 258 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:13:04.808041+00:00 |
| **Runtime** | 292.4s |
| **Parent** | `node_7_0` |
| **Children** | None |
| **Creation Index** | 259 |

---

## Hypothesis

> The 'Adversarial Governance Gap' Hypothesis: Adversarial AI attacks (ATLAS
cases) are significantly more likely to exploit gaps in 'Trust Readiness' (TR)
competencies than 'Integration Readiness' (IR) competencies, indicating that
security vulnerabilities are primarily rooted in trust assurance failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2912 (Likely False) |
| **Surprise** | -0.5409 |
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
| Maybe False | 14.0 |
| Definitely False | 46.0 |

---

## Experiment Plan

**Objective:** Quantify the prevalence of Trust vs. Integration gaps in adversarial incidents.

### Steps
- 1. Load 'step3_incident_coding'.
- 2. Parse the 'competency_gaps' column (comma-separated strings).
- 3. Count occurrences of codes starting with 'TR-' (Trust) vs 'IR-' (Integration).
- 4. Perform a Chi-Square Goodness of Fit test (assuming equal expectation) or a Binomial test to check for significant deviation.

### Deliverables
- Frequency count of TR vs IR gaps; Statistical test results.

---

## Analysis

The experiment was successfully executed on the third attempt after correctly
identifying the 'competency_domains' column for analysis. Processing 52
adversarial case studies yielded a total of 297 competency gaps.

1.  **Findings**: The analysis identified 160 'Trust Readiness' (TR) gaps and
137 'Integration Readiness' (IR) gaps. While TR gaps were more frequent (53.9%
vs 46.1%), the difference was not overwhelming.

2.  **Statistical Significance**: The Chi-Square Goodness of Fit test produced a
statistic of 1.7811 and a **p-value of 0.1820**. Since the p-value exceeds the
standard alpha of 0.05, we fail to reject the null hypothesis of equal
distribution.

3.  **Conclusion (Hypothesis Rejected)**: The data **does not support** the
hypothesis that adversarial attacks are *significantly* more likely to exploit
Trust gaps than Integration gaps. Instead, the results suggest a balanced threat
landscape where attackers exploit vulnerabilities in model trust (e.g., evasion,
inversion) and system integration (e.g., supply chain, access control) with
comparable frequency. Effective defense requires dual focus on both bundles.

---

## Review

The experiment was successfully executed on the third attempt after correctly
identifying the 'competency_domains' column as the source of granular gap data.
The analysis processed 52 adversarial case studies, yielding a total of 297
competency gaps.

**Findings:**
1.  **Frequency:** The analysis identified 160 'Trust Readiness' (TR) gaps and
137 'Integration Readiness' (IR) gaps. While TR gaps were slightly more frequent
(53.9% vs 46.1%), the disparity was not large.
2.  **Statistical Significance:** The Chi-Square Goodness of Fit test yielded a
statistic of 1.7811 and a **p-value of 0.1820**. Since the p-value exceeds the
standard alpha of 0.05, the difference is not statistically significant.

**Conclusion:** The results **fail to support** the 'Adversarial Governance Gap'
hypothesis. There is no statistically significant evidence that adversarial
attacks disproportionately exploit Trust Readiness gaps over Integration
Readiness gaps. Instead, the data suggests a balanced threat landscape where
attackers target vulnerabilities in model robustness (Trust) and system
security/integration (Integration) with comparable frequency. Future governance
frameworks should therefore maintain a dual focus rather than prioritizing one
bundle over the other based on attack prevalence.

---

## Code

```python
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

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Adversarial Governance Gap experiment (Attempt 3)...
Dataset loaded successfully.
Subset shape: (52, 196)
Rows with valid competency_domains: 52

Counts:
Trust Readiness (TR): 160
Integration Readiness (IR): 137

Chi-Square Statistic: 1.7811
P-value: 1.8201e-01
Result: No statistically significant difference between TR and IR prevalence.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot is designed to compare the frequency of two distinct categories of competency gaps ("Trust Readiness" vs. "Integration Readiness") within the context of Adversarial Governance.

### 2. Axes
*   **X-Axis:**
    *   **Labels:** The axis represents categorical data with two specific groups: **"Trust Readiness"** and **"Integration Readiness"**.
    *   **Range:** N/A (Categorical).
*   **Y-Axis:**
    *   **Title:** "Frequency of Competency Gaps".
    *   **Units:** Count/Frequency (implied integer counts).
    *   **Range:** The axis is scaled from **0 to 160**, with tick marks at intervals of 20 (0, 20, 40, ..., 160).

### 3. Data Trends
*   **Tallest Bar:** The blue bar representing **"Trust Readiness"** is the tallest, reaching a value of **160**.
*   **Shortest Bar:** The orange bar representing **"Integration Readiness"** is the shorter of the two, with a value of **137**.
*   **Pattern:** There is a notable disparity between the two categories, with "Trust Readiness" gaps occurring more frequently than "Integration Readiness" gaps. The difference between the two values is 23 units.

### 4. Annotations and Legends
*   **Value Annotations:** Exact frequency counts are annotated directly on top of each bar (**160** for Trust Readiness and **137** for Integration Readiness) to provide precise reading of the data without needing to estimate from the y-axis.
*   **Grid Lines:** Horizontal dashed grid lines are included at every major y-axis tick (intervals of 20) to aid in visual estimation of bar height.
*   **Color Coding:** The bars are distinct in color (Blue for Trust Readiness, Orange for Integration Readiness), though the x-axis labels serve as the primary legend/identifier.

### 5. Statistical Insights
*   **Prevalence of Trust Issues:** The data indicates that "Trust Readiness" is the primary bottleneck or area of failure in this dataset. With 160 reported gaps, it is approximately **16.8% more frequent** than "Integration Readiness" gaps (137).
*   **Gap Analysis:** In the context of "Adversarial Governance," this suggests that stakeholders or systems struggle more significantly with issues related to trust (likely involving reliability, transparency, or confidence) than they do with the mechanics of integration.
*   **Research Implication:** Remediation efforts regarding competency gaps should prioritize "Trust Readiness," as it represents the larger volume of failures. However, "Integration Readiness" remains a significant factor and is not negligible.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
