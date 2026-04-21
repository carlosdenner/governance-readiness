# Experiment 20: node_3_7

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_7` |
| **ID in Run** | 20 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:04:16.930018+00:00 |
| **Runtime** | 280.2s |
| **Parent** | `node_2_7` |
| **Children** | `node_4_9`, `node_4_20` |
| **Creation Index** | 21 |

---

## Hypothesis

> The 'Exfiltration' tactic is uniquely associated with 'Integration Readiness'
gaps (lack of egress filtering), whereas 'Reconnaissance' is independent of
competency bundle, indicating it is a universal precursor.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6129 (Maybe True) |
| **Posterior** | 0.1942 (Likely False) |
| **Surprise** | -0.4859 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 22.0 |
| Uncertain | 0.0 |
| Maybe False | 8.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 18.0 |
| Definitely False | 72.0 |

---

## Experiment Plan

**Objective:** Validate the bundle-specificity of distinct adversarial tactics.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Parse the 'tactics_used' column (semicolon-separated).
- 3. Create a binary indicator for 'Exfiltration' and 'Reconnaissance' for each incident.
- 4. For each tactic, perform a T-test or Chi-square to see if its presence predicts a specific 'trust_integration_split' (convert split to binary 0/1 for Trust/Integration).
- 5. Compare the p-values to determine if Exfiltration is more strongly linked to Integration than Reconnaissance.

### Deliverables
- Statistical test comparisons for selected tactics.

---

## Analysis

The experiment successfully correlated adversarial tactics with competency
bundle orientation using the correct MITRE ATLAS IDs (AML.TA0010 for
Exfiltration, AML.TA0002 for Reconnaissance). The hypothesis that 'Exfiltration'
is uniquely associated with 'Integration Readiness' was not supported. The
T-test results showed no statistically significant difference in integration
scores for Exfiltration (p=0.42) or Reconnaissance (p=0.86). Notably, the
Exfiltration group had a mean score of 0.500 with a standard deviation of 0.000,
indicating that every single incident involving Exfiltration was classified as
'Both' (implicating both Trust and Integration competencies). This result
reflects the broader dataset characteristic where 88% of cases (46/52) are
classified as 'Both', suggesting that in practice, these failure modes rarely
isolate to a single domain.

---

## Review

The experiment was successfully executed after correcting the data parsing logic
to use MITRE ATLAS IDs (AML.TA0010, AML.TA0002) instead of text labels. The
analysis faithfully tested the hypothesis that 'Exfiltration' tactics are
uniquely associated with 'Integration Readiness' gaps.

Summary of Findings:
1. **Hypothesis Rejected**: The results did not support the hypothesis.
Exfiltration incidents (n=15) had a mean Integration Score of 0.500 with a
standard deviation of 0.00, indicating that every single incident involving
Exfiltration was classified as 'Both' (Trust & Integration), rather than
shifting toward 'Integration Readiness' (Score 1.0).
2. **Statistical Results**: T-tests compared the integration scores of incidents
with and without the tactics. For Exfiltration, p=0.42; for Reconnaissance,
p=0.86. Neither result was statistically significant.
3. **Dataset Characteristics**: The results reflect the underlying dataset
distribution, where 88% of all cases (46/52) are classified as involving 'Both'
capability bundles. Consequently, specific adversarial tactics like Exfiltration
do not appear to isolate to a single competency domain in this dataset.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Load dataset
file_path = 'step3_incident_coding.csv'
# Handle potential file location differences
if not os.path.exists(file_path):
    if os.path.exists(os.path.join('..', file_path)):
        file_path = os.path.join('..', file_path)

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# --- Preprocessing ---

# 1. Parse Tactics using correct IDs
# AML.TA0010 = Exfiltration
# AML.TA0002 = Reconnaissance

df['tactics_used'] = df['tactics_used'].fillna('').astype(str)

df['has_exfiltration'] = df['tactics_used'].apply(lambda x: 1 if 'AML.TA0010' in x else 0)
df['has_reconnaissance'] = df['tactics_used'].apply(lambda x: 1 if 'AML.TA0002' in x else 0)

print("\n--- Tactic Frequency ---")
print(f"Incidents with Exfiltration (AML.TA0010): {df['has_exfiltration'].sum()}")
print(f"Incidents with Reconnaissance (AML.TA0002): {df['has_reconnaissance'].sum()}")

# 2. Parse Trust/Integration Split
# Mapping: trust-dominant -> 0, both -> 0.5, integration-dominant -> 1
# This creates a continuous 'Integration Orientation' scale.
mapping = {
    'trust-dominant': 0.0,
    'both': 0.5,
    'integration-dominant': 1.0
}

df['integration_score'] = df['trust_integration_split'].map(mapping)

# Filter out any unmapped rows (though previous check showed none)
df = df.dropna(subset=['integration_score'])

# --- Statistical Analysis ---

def analyze_tactic(name, col_name):
    # Split groups
    present = df[df[col_name] == 1]['integration_score']
    absent = df[df[col_name] == 0]['integration_score']
    
    # Check if we have enough data
    if len(present) < 2 or len(absent) < 2:
        print(f"\nSkipping statistical test for {name} due to insufficient sample size (Present: {len(present)}, Absent: {len(absent)}).")
        return None

    # T-test
    t_stat, p_val = stats.ttest_ind(present, absent, equal_var=False)
    
    print(f"\nAnalysis for {name}:")
    print(f"  Mean Integration Score (Present): {present.mean():.3f} (n={len(present)}) | Std: {present.std():.3f}")
    print(f"  Mean Integration Score (Absent):  {absent.mean():.3f} (n={len(absent)}) | Std: {absent.std():.3f}")
    print(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    return {
        'tactic': name,
        'mean_present': present.mean(),
        'mean_absent': absent.mean(),
        'p_value': p_val
    }

results = []
res_exfil = analyze_tactic('Exfiltration', 'has_exfiltration')
if res_exfil: results.append(res_exfil)

res_recon = analyze_tactic('Reconnaissance', 'has_reconnaissance')
if res_recon: results.append(res_recon)

# --- Visualization ---
if results:
    plt.figure(figsize=(10, 6))

    tactics = [r['tactic'] for r in results]
    means_present = [r['mean_present'] for r in results]
    means_absent = [r['mean_absent'] for r in results]
    p_values = [r['p_value'] for r in results]

    x = np.arange(len(tactics))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, means_present, width, label='Tactic Present', color='skyblue')
    rects2 = ax.bar(x + width/2, means_absent, width, label='Tactic Absent', color='lightgray')

    ax.set_ylabel('Mean Integration Score (0=Trust, 1=Integration)')
    ax.set_title('Integration Orientation by Tactic Presence')
    ax.set_xticks(x)
    ax.set_xticklabels(tactics)
    ax.set_ylim(0, 1.0)
    ax.legend()

    # Add p-values and counts
    for i, p in enumerate(p_values):
        height = max(means_present[i], means_absent[i])
        ax.text(i, height + 0.02, f'p={p:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()
else:
    print("No valid results to plot.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: 
--- Tactic Frequency ---
Incidents with Exfiltration (AML.TA0010): 15
Incidents with Reconnaissance (AML.TA0002): 20

Analysis for Exfiltration:
  Mean Integration Score (Present): 0.500 (n=15) | Std: 0.000
  Mean Integration Score (Absent):  0.527 (n=37) | Std: 0.202
  T-test: t=-0.813, p=0.4217

Analysis for Reconnaissance:
  Mean Integration Score (Present): 0.525 (n=20) | Std: 0.197
  Mean Integration Score (Absent):  0.516 (n=32) | Std: 0.155
  T-test: t=0.181, p=0.8576

STDERR:
/usr/local/lib/python3.13/site-packages/scipy/stats/_axis_nan_policy.py:592: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
  res = hypotest_fun_out(*samples, **kwds)


=== Plot Analysis (figure 2) ===
Based on the provided image, here is the analysis of the plot:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (or Clustered Bar Plot).
*   **Purpose:** The plot compares the mean values of a specific metric ("Integration Orientation Score") across two main categories ("Exfiltration" and "Reconnaissance"), further split by a binary condition ("Tactic Present" vs. "Tactic Absent").

### 2. Axes
*   **X-axis:**
    *   **Title/Labels:** Categorical labels representing specific tactics: "Exfiltration" and "Reconnaissance".
    *   **Range:** N/A (Categorical).
*   **Y-axis:**
    *   **Title/Label:** "Mean Integration Score (0=Trust, 1=Integration)".
    *   **Units:** Dimensionless score on a scale from 0 to 1.
    *   **Range:** The axis spans from 0.0 to 1.0, with tick marks at 0.2 intervals.

### 3. Data Trends
*   **General Observation:** across all categories and conditions, the Mean Integration Score hovers very close to the midpoint (0.5).
*   **Exfiltration Group:**
    *   The "Tactic Absent" bar (gray) is slightly taller than the "Tactic Present" bar (blue).
    *   The values appear to be roughly 0.50 for "Present" and slightly above 0.52 for "Absent".
*   **Reconnaissance Group:**
    *   The "Tactic Present" bar (blue) is marginally taller than the "Tactic Absent" bar (gray).
    *   Both bars appear to be just above the 0.50 mark.
*   **Comparison:** The variation in bar height between the two conditions (Present vs. Absent) is minimal for both tactic categories.

### 4. Annotations and Legends
*   **Legend:** Located in the top right corner.
    *   **Blue:** Represents "Tactic Present".
    *   **Gray:** Represents "Tactic Absent".
*   **P-values:** Statistical annotations are placed above each group of bars to indicate significance testing results:
    *   Above **Exfiltration**: **p=0.422**
    *   Above **Reconnaissance**: **p=0.858**
*   **Title:** "Integration Orientation by Tactic Presence".

### 5. Statistical Insights
*   **No Statistically Significant Difference:** The p-values provided (0.422 and 0.858) are both well above the standard significance threshold (typically $\alpha = 0.05$). This indicates that there is **no statistically significant difference** in the Mean Integration Score between cases where the tactic is present versus when it is absent.
*   **Neutral Orientation:** Since the scores are centered around 0.5 (on a scale where 0 is Trust and 1 is Integration), the participants or subjects generally displayed a neutral orientation, neither strongly leaning toward Trust nor Integration, regardless of the tactic used.
*   **Conclusion:** The presence or absence of "Exfiltration" or "Reconnaissance" tactics does not appear to influence the Integration Orientation score in a meaningful way based on this data.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
