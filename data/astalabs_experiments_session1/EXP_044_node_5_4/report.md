# Experiment 44: node_5_4

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_4` |
| **ID in Run** | 44 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:53:18.291850+00:00 |
| **Runtime** | 322.0s |
| **Parent** | `node_4_10` |
| **Children** | `node_6_2` |
| **Creation Index** | 45 |

---

## Hypothesis

> Incidents characterized by 'Security' harm types are significantly more likely
to be classified as 'Prevention Failures' compared to Non-Security harms
(Privacy, Reliability, etc.), which are more likely to be Detection/Response
failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2000 (Likely False) |
| **Surprise** | -0.6290 |
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
| Maybe False | 0.0 |
| Definitely False | 84.0 |

---

## Experiment Plan

**Objective:** Investigate the relationship between Harm Type and Failure Mode.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Create a binary variable 'is_security_harm' (True if harm_type == 'security', else False).
- 3. Create a binary variable 'is_prevention_failure' (True if failure_mode == 'prevention_failure', else False).
- 4. Create a contingency table (crosstab) of Harm Category vs Failure Mode.
- 5. Perform a Chi-square test of independence (or Fisher's Exact Test if counts are low).

### Deliverables
- 1. Contingency table.
- 2. Chi-square test results.
- 3. Stacked bar chart showing failure mode proportions.

---

## Analysis

The experiment was successfully executed, utilizing 'step3_incident_coding.csv'
to test the relationship between Harm Type and Failure Mode.

**Descriptive Statistics:**
- **Non-Security Harms:** n=16. 100% (16/16) were Prevention Failures.
- **Security Harms:** n=36. 97.2% (35/36) were Prevention Failures; 2.8% (1/36)
was a Detection/Response Failure.

**Statistical Results:**
- **Method:** Fisher's Exact Test (due to low cell counts).
- **Result:** Odds Ratio = 0.0, p-value = 1.0.
- **Conclusion:** The p-value of 1.0 indicates absolutely no statistically
significant difference between the groups. The null hypothesis cannot be
rejected.

**Visualization:**
The stacked bar chart visually confirms the overwhelming dominance of
'Prevention' failures across both categories. The 'Detection/Response' failure
mode is virtually non-existent in this dataset (occurring only once), making it
impossible to establish any correlation with harm type.

**Hypothesis Evaluation:**
The hypothesis that Security harms are significantly more likely to be
Prevention failures compared to Non-Security harms is **rejected**. In fact,
Prevention failures are the universal dominant mode (51/52 cases) regardless of
harm type.

---

## Review

The experiment pipeline was faithfully executed, successfully loading and
analyzing the complete set of 18 datasets derived from the 5-step agentic
analysis. The summary statistics and hypothesis tests provided a comprehensive
evaluation of the 'Strategic AI Orientation' framework components.

Key Findings:
1. **Dataset Characteristics:** The generated competency statements (Step 2) are
well-balanced between 'Integration Readiness' (n=23) and 'Trust Readiness'
(n=19). However, the validation dataset (Step 3, MITRE ATLAS incidents) is
heavily skewed, with 69% of incidents classified as 'Security' harms and 98%
(51/52) as 'Prevention Failures'.

2. **Hypothesis Testing Results:** All three specific hypotheses tested were
rejected based on the data:
   - **Evidence vs. Complexity:** The architectural complexity (number of
controls) of a competency is independent of its evidence confidence level
(p=0.93).
   - **Incident Complexity:** Multi-domain failures (involving both Trust and
Integration gaps) do not exhibit significantly higher attack sophistication
(technique counts) than single-domain failures (p=0.81).
   - **Harm vs. Failure Mode:** There is no statistically significant
relationship between Harm Type and Failure Mode (p=1.0) because the dataset is
almost exclusively composed of prevention failures, regardless of whether the
harm is Security or Non-Security.

3. **Implications:** The analysis suggests that while the framework's
definitions are structurally sound and balanced, the validation data (MITRE
ATLAS) is highly homogeneous regarding failure mechanisms. The 'Trust Readiness'
and 'Integration Readiness' bundles appear to be distinct but structurally
equivalent in terms of complexity and supporting evidence strength.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Load the dataset
filename = 'step3_incident_coding.csv'
# Try current directory first, then parent
if os.path.exists(filename):
    file_path = filename
elif os.path.exists(f'../{filename}'):
    file_path = f'../{filename}'
else:
    raise FileNotFoundError(f"{filename} not found")

df = pd.read_csv(file_path)

# Prepare data for analysis
# Group Harm Types: Security vs Non-Security
df['harm_category'] = df['harm_type'].apply(lambda x: 'Security' if str(x).strip().lower() == 'security' else 'Non-Security')

# Group Failure Modes: Prevention vs Detection/Response
# Note: Metadata indicates 51/52 are prevention failures
df['failure_category'] = df['failure_mode'].apply(lambda x: 'Prevention' if str(x).strip().lower() == 'prevention_failure' else 'Detection/Response')

# Create Contingency Table
contingency_table = pd.crosstab(df['harm_category'], df['failure_category'])
print("=== Contingency Table (Harm Category vs Failure Mode) ===")
print(contingency_table)

# Statistical Testing
# Using Fisher's Exact Test due to expected low counts in cells
if contingency_table.shape == (2, 2):
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    print(f"\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")
else:
    print("\n[Info] Contingency table dimensions are not 2x2. One category might be missing from the data.")
    # If the table is 2x1 (e.g., only Prevention exists for one group), fill missing col with 0 for display
    if contingency_table.shape[1] == 1:
        print("All observed failures fall into a single category.")

# Visualization
# Normalize to show proportions
props = pd.crosstab(df['harm_category'], df['failure_category'], normalize='index')

fig, ax = plt.subplots(figsize=(8, 6))
props.plot(kind='bar', stacked=True, ax=ax, color=['#ff9999', '#66b3ff'])

plt.title('Proportion of Failure Modes by Harm Category')
plt.xlabel('Harm Category')
plt.ylabel('Proportion')
plt.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Contingency Table (Harm Category vs Failure Mode) ===
failure_category  Detection/Response  Prevention
harm_category                                   
Non-Security                       0          16
Security                           1          35

Fisher's Exact Test Results:
Odds Ratio: 0.0000
P-value: 1.0000


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart (specifically normalized to show proportions).
*   **Purpose:** The plot compares the relative distribution of different "Failure Modes" within two distinct "Harm Categories." By normalizing the height of the bars to 1.0 (or 100%), it focuses on the composition ratio rather than absolute counts.

### 2. Axes
*   **X-axis:**
    *   **Label:** "Harm Category"
    *   **Categories:** Two discrete categories are listed: "Non-Security" and "Security."
*   **Y-axis:**
    *   **Label:** "Proportion"
    *   **Range:** The axis ranges from 0.0 to just over 1.0, with tick marks at 0.0, 0.2, 0.4, 0.6, 0.8, and 1.0.
    *   **Units:** The values represent a ratio or probability (0 to 1), where 1.0 is equivalent to 100%.

### 3. Data Trends
*   **Non-Security Category:**
    *   The bar is entirely blue.
    *   This indicates that 100% of the failures in the "Non-Security" category are attributed to the "Prevention" failure mode. There is no visible red segment for "Detection/Response."
*   **Security Category:**
    *   The bar is overwhelmingly blue but contains a small red segment at the bottom.
    *   This indicates that while the vast majority (visually estimated at >95%) of failures are "Prevention" based, there is a small, non-zero proportion of failures attributed to "Detection/Response."

### 4. Annotations and Legends
*   **Plot Title:** "Proportion of Failure Modes by Harm Category" — Clearly states the subject of the visualization.
*   **Legend:** Located in the upper right corner with the title **"Failure Mode."**
    *   **Light Red/Pink:** Represents "Detection/Response."
    *   **Light Blue:** Represents "Prevention."

### 5. Statistical Insights
*   **dominance of Prevention Failures:** Across both harm categories (Security and Non-Security), "Prevention" is the overwhelmingly dominant failure mode. This suggests that the primary issue in both contexts is a failure to prevent the harm from occurring in the first place, rather than a failure to detect or respond to it.
*   **Exclusivity in Non-Security:** For Non-Security harms, the data suggests that "Detection/Response" failures are effectively non-existent in this dataset. Every recorded instance falls under "Prevention."
*   **Comparison of Complexity:** The "Security" category is slightly more complex than "Non-Security," as it involves a secondary failure mode ("Detection/Response"), whereas "Non-Security" is monolithic. However, the contribution of "Detection/Response" remains marginal.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
