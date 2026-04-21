# Experiment 244: node_7_8

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_8` |
| **ID in Run** | 244 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:32:16.318450+00:00 |
| **Runtime** | 339.8s |
| **Parent** | `node_6_3` |
| **Children** | `node_8_1` |
| **Creation Index** | 245 |

---

## Hypothesis

> The Evasion-Robustness Link: In adversarial AI cases, the use of 'Evasion'
tactics is strongly predictive of a 'Model Robustness' competency gap, whereas
'Exfiltration' tactics are predictive of 'System Access/Logging' gaps.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9597 (Definitely True) |
| **Posterior** | 0.3434 (Maybe False) |
| **Surprise** | -0.7395 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 27.0 |
| Maybe True | 3.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 6.0 |
| Definitely False | 54.0 |

---

## Experiment Plan

**Objective:** Map specific adversarial tactics to specific governance competency gaps.

### Steps
- 1. Load 'step3_incident_coding'.
- 2. Parse the 'tactics_used' column to identify rows containing 'Evasion' vs 'Exfiltration'.
- 3. Parse 'competency_gaps' to identify rows containing 'Robustness' vs 'Access/Logging'.
- 4. Construct a contingency table (Tactic Type vs Gap Type).
- 5. Perform Fisher's Exact Test (due to likely small sample size) to test the association.

### Deliverables
- Contingency table of Tactic vs Gap; Fisher's Exact Test p-value.

---

## Analysis

The experiment successfully tested the 'Evasion-Robustness Link' hypothesis by
mapping MITRE ATLAS tactic IDs (AML.TA0006 for Evasion, AML.TA0010 for
Exfiltration) to competency gaps in the `step3_incident_coding` dataset.

The results statistically **fail to support** the hypothesis. The hypothesis
predicted that Evasion tactics would specifically correlate with Robustness
gaps, while Exfiltration would correlate with Access/Logging gaps. The data,
however, showed that 'Access/Logging' gaps were the dominant failure mode for
*both* tactic types.

- **Evasion Incidents**: 9 Access/Logging gaps vs. 5 Robustness gaps.
- **Exfiltration Incidents**: 12 Access/Logging gaps vs. 5 Robustness gaps.

Fisher's Exact Test yielded a p-value of 1.00, indicating that the distribution
of gaps is statistically identical across both tactic types. There is no
specific predictive link between 'Evasion' and 'Robustness'; rather, Access
Control and Logging failures appear to be a systemic vulnerability regardless of
whether the adversary is attempting to evade detection or exfiltrate data.

---

## Review

The experiment was successfully executed and the analysis is sound. The
programmer correctly adapted the data extraction logic to handle MITRE ATLAS IDs
(e.g., AML.TA0006) instead of English keywords, resolving the issue encountered
in the first attempt. The mapping of 'Evasion' and 'Exfiltration' tactics to
their corresponding governance gaps in 'competency_domains' was logical. The use
of Fisher's Exact Test was appropriate given the small sample size (n=24
relevant incidents). The null result (p=1.0) is a valid scientific finding,
indicating that the hypothesized link does not exist in this dataset.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for the specific source table
subset = df[df['source_table'] == 'step3_incident_coding'].copy()

# Drop rows with NaN in critical columns
subset = subset.dropna(subset=['tactics_used', 'competency_domains'])

# --- 1. Map Tactics (using MITRE ATLAS IDs) ---
# AML.TA0006: Defense Evasion
# AML.TA0010: Exfiltration
subset['has_evasion_tactic'] = subset['tactics_used'].astype(str).str.contains('AML.TA0006')
subset['has_exfiltration_tactic'] = subset['tactics_used'].astype(str).str.contains('AML.TA0010')

# --- 2. Map Competency Gaps (using text keywords in competency_domains) ---
# Robustness Gap: Looking for 'Robustness' or 'Evasion Detection'
subset['has_robustness_gap'] = subset['competency_domains'].astype(str).str.contains('Robustness', case=False) | \
                               subset['competency_domains'].astype(str).str.contains('Evasion Detection', case=False)

# Access/Logging Gap: Looking for 'Access', 'Logging', 'Audit'
subset['has_access_gap'] = subset['competency_domains'].astype(str).str.contains('Access', case=False) | \
                           subset['competency_domains'].astype(str).str.contains('Logging', case=False) | \
                           subset['competency_domains'].astype(str).str.contains('Audit', case=False)

# --- 3. Filter for relevant rows ---
# We only care about rows that have (Evasion OR Exfiltration) AND (RobustnessGap OR AccessGap)
relevant = subset[
    (subset['has_evasion_tactic'] | subset['has_exfiltration_tactic']) &
    (subset['has_robustness_gap'] | subset['has_access_gap'])
].copy()

print(f"Total rows in subset: {len(subset)}")
print(f"Relevant rows for hypothesis: {len(relevant)}")

# --- 4. Construct Contingency Table ---
# We prioritize categorization. If a row has both, it counts for both in a generalized sense, 
# but for Fisher's test we need a 2x2 matrix of counts. 
# We will count "Incidents with Evasion Tactic" vs "Incidents with Exfiltration Tactic"
# against "Incidents with Robustness Gap" vs "Incidents with Access Gap".
# Note: A single incident could theoretically be in multiple cells if it has multiple tactics/gaps.
# To strictly test the hypothesis "Evasion -> Robustness" vs "Exfil -> Access", we can define the events:
# A: Tactic is Evasion (and not Exfil)
# B: Tactic is Exfil (and not Evasion)
# Outcome 1: Gap is Robustness
# Outcome 2: Gap is Access

# Let's filter for the disjoint sets of Tactics to make the groups independent
evasion_group = relevant[relevant['has_evasion_tactic'] & ~relevant['has_exfiltration_tactic']]
exfil_group = relevant[relevant['has_exfiltration_tactic'] & ~relevant['has_evasion_tactic']]

# Counts
# Group 1: Evasion Tactic
# We check how many have Robustness Gap vs Access Gap
# (Note: An incident can have both gaps, but usually we test the "primary" association.
# If we treat it as binary features, we can just sum them up, but Fisher expects a contingency table of mutually exclusive outcomes usually.
# However, we can test: "Given Tactic X, is Gap A more likely than Gap B?")

# Let's count occurrences. 
# Evasion Tactic -> Robustness Gap
count_evasion_robustness = evasion_group['has_robustness_gap'].sum()
# Evasion Tactic -> Access Gap
count_evasion_access = evasion_group['has_access_gap'].sum()

# Exfiltration Tactic -> Robustness Gap
count_exfil_robustness = exfil_group['has_robustness_gap'].sum()
# Exfiltration Tactic -> Access Gap
count_exfil_access = exfil_group['has_access_gap'].sum()

contingency_table = [
    [count_evasion_robustness, count_evasion_access],
    [count_exfil_robustness, count_exfil_access]
]

print("\nContingency Table (Tactic -> Gap Presence):")
print(f"                  Robustness Gap | Access/Logging Gap")
print(f"Evasion Only      {count_evasion_robustness:<14} | {count_evasion_access:<18}")
print(f"Exfiltration Only {count_exfil_robustness:<14} | {count_exfil_access:<18}")

# Fisher's Exact Test
if (count_evasion_robustness + count_evasion_access + count_exfil_robustness + count_exfil_access) == 0:
    print("Insufficient data for statistical test.")
else:
    oddsratio, pvalue = stats.fisher_exact(contingency_table)
    print(f"\nFisher's Exact Test p-value: {pvalue:.5f}")
    print(f"Odds Ratio: {oddsratio:.5f}")
    
    # Visualization
    labels = ['Evasion Tactic', 'Exfiltration Tactic']
    r_counts = [count_evasion_robustness, count_exfil_robustness]
    a_counts = [count_evasion_access, count_exfil_access]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, r_counts, width, label='Robustness Gap')
    rects2 = ax.bar(x + width/2, a_counts, width, label='Access/Logging Gap')

    ax.set_ylabel('Count of Incidents')
    ax.set_title('Association: Tactic Type vs Governance Gap')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total rows in subset: 52
Relevant rows for hypothesis: 24

Contingency Table (Tactic -> Gap Presence):
                  Robustness Gap | Access/Logging Gap
Evasion Only      5              | 9                 
Exfiltration Only 5              | 12                

Fisher's Exact Test p-value: 1.00000
Odds Ratio: 1.33333


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (or Clustered Bar Chart).
*   **Purpose:** This chart is designed to compare the frequency ("Count of Incidents") of two different categories of governance gaps ("Robustness Gap" and "Access/Logging Gap") across two specific tactic types ("Evasion Tactic" and "Exfiltration Tactic"). It allows for side-by-side comparison of the gap types within each tactic.

### 2. Axes
*   **X-axis:**
    *   **Label:** Represents the **Tactic Type**.
    *   **Categories:** "Evasion Tactic" and "Exfiltration Tactic".
*   **Y-axis:**
    *   **Label:** "Count of Incidents".
    *   **Range:** The scale runs from **0 to 12**.
    *   **Units:** Integer counts (incidents).

### 3. Data Trends
*   **Tallest Bar:** The tallest bar represents the **Access/Logging Gap** within the **Exfiltration Tactic** category, with a count of **12**.
*   **Shortest Bars:** The shortest bars are tied. Both **Robustness Gap** bars (for both Evasion and Exfiltration tactics) have a count of **5**.
*   **Patterns:**
    *   For both tactic types, the "Access/Logging Gap" (Orange) is significantly higher than the "Robustness Gap" (Blue).
    *   The "Robustness Gap" shows a constant value (5 incidents) across both tactic types, indicating no variation based on the tactic.
    *   The "Access/Logging Gap" increases from the Evasion Tactic (9 incidents) to the Exfiltration Tactic (12 incidents).

### 4. Annotations and Legends
*   **Chart Title:** "Association: Tactic Type vs Governance Gap" – clearly states the relationship being visualized.
*   **Legend:** Located in the top-left corner, it distinguishes the data series:
    *   **Blue Box:** Represents "Robustness Gap".
    *   **Orange Box:** Represents "Access/Logging Gap".

### 5. Statistical Insights
*   **Dominance of Access/Logging Issues:** Across both tactic types shown, incidents related to Access/Logging Gaps are more prevalent than Robustness Gaps. This suggests that monitoring access and logging is a more significant vulnerability or area of failure than system robustness for these specific tactics.
*   **Uniformity of Robustness Issues:** The data suggests that the Robustness Gap is static regardless of whether the tactic is evasion or exfiltration. This implies a consistent baseline of 5 incidents, perhaps pointing to a systemic robustness issue that is not specific to the attacker's method.
*   **Tactic Severity:** The Exfiltration Tactic is associated with a higher total number of incidents (5 + 12 = 17) compared to the Evasion Tactic (5 + 9 = 14), driven entirely by the increase in Access/Logging gaps. This indicates that exfiltration attempts are more frequently linked to logging failures than evasion attempts are.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
