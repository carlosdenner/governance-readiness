# Experiment 78: node_3_19

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_19` |
| **ID in Run** | 78 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:01:51.140890+00:00 |
| **Runtime** | 206.4s |
| **Parent** | `node_2_6` |
| **Children** | None |
| **Creation Index** | 79 |

---

## Hypothesis

> Sub-competencies in the 'Trust Readiness' bundle exhibit higher 'Harm Entropy'
(diversity of resulting harm types) than 'Integration Readiness' sub-
competencies, which are more narrowly focused on security harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7396 (Likely True) |
| **Posterior** | 0.3136 (Maybe False) |
| **Surprise** | -0.4944 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 23.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 72.0 |
| Definitely False | 18.0 |

---

## Experiment Plan

**Objective:** Compare the diversity of harm types associated with each bundle.

### Steps
- 1. Load 'step3_coverage_map.csv'.
- 2. Parse the 'primary_harm_types' column (e.g., 'security(16); privacy(2)') to extract counts for each harm type per sub-competency.
- 3. Calculate the Shannon Entropy for each sub-competency based on its harm type distribution.
- 4. Split the sub-competencies by 'bundle'.
- 5. Compare the mean Entropy values using a t-test.
- 6. Visualize entropy scores by sub-competency using a bar chart.

### Deliverables
- Entropy calculation logic, t-test results comparing bundle means, and bar chart.

---

## Analysis

The experiment successfully tested the hypothesis that 'Trust Readiness' sub-
competencies exhibit higher harm entropy (diversity of harm types) than
'Integration Readiness' ones. The code successfully parsed the
'primary_harm_types' field from 'step3_coverage_map.csv', calculated Shannon
Entropy for all 16 sub-competencies, and performed an independent t-test.

**Findings:**
1.  **Descriptive Statistics:** The results contradicted the hypothesis
directionally. 'Integration Readiness' sub-competencies actually had a higher
mean entropy (0.4835) compared to 'Trust Readiness' (0.3084).
2.  **Statistical Significance:** The t-test yielded a p-value of 0.3779 (t =
-0.9114), indicating that the difference between the two groups is **not
statistically significant**.
3.  **Visual Insights:** The generated bar chart reveals that entropy is driven
by specific, high-activity sub-competencies (e.g., 'Monitoring & Detection' in
Trust, 'Orchestration' and 'Tool-Use Boundaries' in Integration). A significant
number of sub-competencies in both bundles have zero entropy, likely due to
having few mapped incidents or incidents of a single type (e.g., purely
'security'), which contributed to high standard deviations and the lack of
statistical significance.

**Conclusion:** The hypothesis is rejected. Harm diversity appears to be a
function of specific sub-competency exposure rather than the high-level 'Trust'
vs 'Integration' classification.

---

## Review

The experiment successfully tested the hypothesis that 'Trust Readiness' sub-
competencies exhibit higher harm entropy (diversity of harm types) than
'Integration Readiness' ones. The code successfully parsed the
'primary_harm_types' field from 'step3_coverage_map.csv', calculated Shannon
Entropy for all 16 sub-competencies, and performed an independent t-test.

**Findings:**
1.  **Descriptive Statistics:** The results contradicted the hypothesis
directionally. 'Integration Readiness' sub-competencies actually had a higher
mean entropy (0.4835) compared to 'Trust Readiness' (0.3084).
2.  **Statistical Significance:** The t-test yielded a p-value of 0.3779 (t =
-0.9114), indicating that the difference between the two groups is **not
statistically significant**.
3.  **Visual Insights:** The generated bar chart reveals that entropy is driven
by specific, high-activity sub-competencies (e.g., 'Monitoring & Detection' in
Trust, 'Orchestration' and 'Tool-Use Boundaries' in Integration). A significant
number of sub-competencies in both bundles have zero entropy, likely due to
having few mapped incidents or incidents of a single type (e.g., purely
'security'), which contributed to high standard deviations and the lack of
statistical significance.

**Conclusion:** The hypothesis is rejected. Harm diversity appears to be a
function of specific sub-competency exposure rather than the high-level 'Trust'
vs 'Integration' classification.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re
import math
import os

def parse_harm_types(harm_str):
    """
    Parses a string like 'security(16); privacy(2)' into a dictionary.
    """
    if pd.isna(harm_str) or str(harm_str).strip() == "":
        return {}
    # Regex to capture name (allowing letters, numbers, underscores, spaces) and count
    pattern = r"([a-zA-Z0-9_\s]+)\((\d+)\)"
    matches = re.findall(pattern, str(harm_str))
    return {name.strip(): int(count) for name, count in matches}

def calculate_entropy(counts):
    """
    Calculates Shannon Entropy based on count dictionary.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy

def run_experiment():
    # 1. Load Dataset
    file_path = '../step3_coverage_map.csv'
    if not os.path.exists(file_path):
        # Fallback to local if parent directory check fails (e.g., if env structure differs)
        file_path = 'step3_coverage_map.csv'
        if not os.path.exists(file_path):
            print("Error: step3_coverage_map.csv not found.")
            return

    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # 2. Parse Harm Types
    df['harm_counts'] = df['primary_harm_types'].apply(parse_harm_types)

    # 3. Calculate Entropy
    df['entropy'] = df['harm_counts'].apply(calculate_entropy)

    # 4. Split by Bundle
    trust_mask = df['bundle'].str.contains('Trust', case=False, na=False)
    integration_mask = df['bundle'].str.contains('Integration', case=False, na=False)

    trust_df = df[trust_mask].copy()
    integration_df = df[integration_mask].copy()

    # 5. Statistical Test
    t_stat, p_val = stats.ttest_ind(trust_df['entropy'], integration_df['entropy'], equal_var=False)
    
    print("\n=== Harm Entropy Statistics ===")
    print(f"Trust Readiness (n={len(trust_df)}):")
    print(f"  Mean Entropy: {trust_df['entropy'].mean():.4f}")
    print(f"  Std Dev:      {trust_df['entropy'].std():.4f}")
    print(f"Integration Readiness (n={len(integration_df)}):")
    print(f"  Mean Entropy: {integration_df['entropy'].mean():.4f}")
    print(f"  Std Dev:      {integration_df['entropy'].std():.4f}")
    
    print("\n=== Hypothesis Test ===")
    print(f"Independent t-test results: t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("Conclusion: Significant difference in Harm Entropy between bundles.")
    else:
        print("Conclusion: No significant difference in Harm Entropy between bundles.")

    # 6. Visualization
    plt.figure(figsize=(14, 7))
    
    # Concatenate for plotting, Trust first then Integration
    plot_df = pd.concat([trust_df, integration_df])
    
    # Color mapping: Trust = SkyBlue, Integration = Salmon
    colors = ['skyblue' if 'Trust' in b else 'salmon' for b in plot_df['bundle']]
    
    # Create Bar Chart
    bars = plt.bar(plot_df['sub_competency_name'], plot_df['entropy'], color=colors, edgecolor='grey')
    
    plt.title('Harm Diversity (Shannon Entropy) by Sub-Competency')
    plt.ylabel('Shannon Entropy (nats)')
    plt.xlabel('Sub-Competency')
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='grey', label='Trust Readiness'),
        Patch(facecolor='salmon', edgecolor='grey', label='Integration Readiness')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()

    # Debug output of the dataframe to verify parsing if needed
    # print(df[['sub_competency_id', 'primary_harm_types', 'entropy']])

if __name__ == "__main__":
    run_experiment()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_coverage_map.csv

=== Harm Entropy Statistics ===
Trust Readiness (n=8):
  Mean Entropy: 0.3084
  Std Dev:      0.3523
Integration Readiness (n=8):
  Mean Entropy: 0.4835
  Std Dev:      0.4140

=== Hypothesis Test ===
Independent t-test results: t=-0.9114, p=0.3779
Conclusion: No significant difference in Harm Entropy between bundles.


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot compares the "Harm Diversity" (measured by Shannon Entropy) across various distinct categories labeled as "Sub-Competencies." It distinguishes between two overarching groups: "Trust Readiness" and "Integration Readiness."

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Sub-Competency"
    *   **Labels:** Categorical text labels representing specific technical or governance areas (e.g., "Risk Policy & Accountability," "Monitoring & Detection," "Orchestration & Execution Controls"). The labels are rotated 45 degrees to accommodate the text length.
*   **Y-Axis:**
    *   **Title:** "Shannon Entropy (nats)"
    *   **Range:** The scale runs from 0.0 to roughly 0.9 (the highest tick mark is 0.8, but several bars exceed this line).
    *   **Units:** Nats (natural units of information).

### 3. Data Trends
*   **Trust Readiness (Blue Bars):**
    *   **Highest Value:** "Monitoring & Detection" is the distinct outlier in this group, showing the highest entropy (approx. 0.9 nats).
    *   **Moderate Values:** "Risk Policy & Accountability," "Incident Response & Recovery," and "Supply Chain & Third-Party Risk" all hover around the 0.5 to 0.55 range.
    *   **Zero/Missing Values:** Several categories, such as "Threat Mapping & Reconnaissance Defense" and "Data Governance & Exfiltration Prevention," have no visible bars, indicating a Shannon Entropy of zero.
*   **Integration Readiness (Red/Salmon Bars):**
    *   **Highest Values:** "Orchestration & Execution Controls" and "Tool-Use Boundaries & Access Control" are tied for the highest values on the entire chart (approx. 0.9 nats).
    *   **High/Moderate Values:** "Modular Architecture & Resource Controls" (~0.8) and "RAG Architecture & Data Grounding" (~0.7) are also relatively high compared to the average. "Nondeterminism Management" sits lower at roughly 0.6.
    *   **Zero/Missing Values:** Categories like "GenAIOps / MLOps Lifecycle" and "HITL Architecture Patterns" show no visible data.

### 4. Annotations and Legends
*   **Legend:** Located in the top right corner, identifying the color coding:
    *   **Light Blue:** Trust Readiness.
    *   **Salmon/Red:** Integration Readiness.
*   **Title:** "Harm Diversity (Shannon Entropy) by Sub-Competency" sits at the top center.

### 5. Statistical Insights
*   **Distribution of Harm Diversity:** "Integration Readiness" sub-competencies generally exhibit higher harm diversity than "Trust Readiness" categories. Four out of the five visible red bars exceed 0.6 nats, whereas only one blue bar exceeds that threshold.
*   **Concentration of Risk/Variance:** High Shannon entropy indicates a high diversity of harm types. This suggests that the areas of **Monitoring & Detection**, **Orchestration & Execution Controls**, and **Tool-Use Boundaries** are susceptible to a wider variety of distinct harm types compared to areas like "Risk Policy," which likely face fewer or more uniform types of harm.
*   **Sparse Data:** A significant portion of the sub-competencies (nearly half) show zero entropy. This implies either a lack of data for these specific competencies or that the harms identified in these areas are completely uniform (zero diversity).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
