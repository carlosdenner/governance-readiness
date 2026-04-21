# Experiment 65: node_4_28

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_28` |
| **ID in Run** | 65 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:05:50.327945+00:00 |
| **Runtime** | 256.0s |
| **Parent** | `node_3_5` |
| **Children** | `node_5_30`, `node_5_50` |
| **Creation Index** | 66 |

---

## Hypothesis

> The 'Theory-Reality' Gap: The frequency of 'Evasion' tactics in research-
oriented datasets (ATLAS) is significantly higher than the frequency of real-
world 'Robustness' failures reported in incident databases (AIID), suggesting a
disconnect between academic threat models and actual failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9194 (Definitely True) |
| **Posterior** | 0.9725 (Definitely True) |
| **Surprise** | +0.0638 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 22.0 |
| Maybe True | 8.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Quantify the disparity between theoretical adversarial tactics and real-world failure modes.

### Steps
- 1. Load `step3_incident_coding` (ATLAS context) and `aiid_incidents`.
- 2. In ATLAS: Calculate proportion of cases containing 'Evasion' or 'AML.TA000X' tactics.
- 3. In AIID: Calculate proportion of incidents where `Known AI Technical Failure` maps to 'Robustness' or 'Reliability'.
- 4. Perform a Two-Sample Z-test for proportions to compare the ATLAS Evasion rate vs. AIID Robustness rate.

### Deliverables
- Comparison bar chart, proportions for each dataset, and Z-test statistics.

---

## Analysis

The experiment was successfully executed and strongly validates the 'Theory-
Reality Gap' hypothesis.

1. **Data Parsing**: The script successfully loaded the sparse dataset and
identified the relevant subsets. It correctly extracted 52 ATLAS cases and 1,362
AIID incidents.

2. **Disparity Findings**:
   - **Theoretical Context (ATLAS)**: 'Evasion' tactics are highly prevalent,
appearing in **63.5%** (33/52) of the adversarial case studies.
   - **Real-World Context (AIID)**: In stark contrast, 'Robustness' or
'Reliability' failures are virtually non-existent in the incident database,
accounting for only **0.3%** (4/1,362) of reported failures.

3. **Statistical Significance**: The Two-Sample Z-test yielded a Z-score of
**28.01** and a p-value of **1.39e-172**, confirming that this discrepancy is
statistically significant to an extreme degree.

4. **Conclusion**: The results quantify a massive disconnect. While the security
research community focuses heavily on evasion (e.g., adversarial examples),
current real-world AI failures are almost never attributed to these technical
robustness issues, suggesting a misalignment between threat modeling priorities
and observed deployment risks.

---

## Review

The experiment was successfully executed and strongly validates the 'Theory-
Reality Gap' hypothesis.

1. **Data Parsing**: The script successfully loaded the sparse dataset and
identified the relevant subsets. It correctly extracted 52 ATLAS cases and 1,362
AIID incidents.

2. **Disparity Findings**:
   - **Theoretical Context (ATLAS)**: 'Evasion' tactics are highly prevalent,
appearing in **63.5%** (33/52) of the adversarial case studies.
   - **Real-World Context (AIID)**: In stark contrast, 'Robustness' or
'Reliability' failures are virtually non-existent in the incident database,
accounting for only **0.3%** (4/1,362) of reported failures.

3. **Statistical Significance**: The Two-Sample Z-test yielded a Z-score of
**28.01** and a p-value of **1.39e-172**, confirming that this discrepancy is
statistically significant to an extreme degree.

4. **Conclusion**: The results quantify a massive disconnect. While the security
research community focuses heavily on evasion (e.g., adversarial examples),
current real-world AI failures are almost never attributed to these technical
robustness issues, suggesting a misalignment between threat modeling priorities
and observed deployment risks.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
import os
import sys
import traceback

# [debug]
# print("Starting experiment...")

def load_data():
    # Try current directory first, then parent
    candidates = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
    for path in candidates:
        if os.path.exists(path):
            print(f"Found dataset at: {path}")
            return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError("Could not find astalabs_discovery_all_data.csv in current or parent directory.")

try:
    # Load the dataset
    df = load_data()

    # Segment the data
    atlas_df = df[df['source_table'] == 'step3_incident_coding'].copy()
    aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

    print(f"ATLAS subset shape: {atlas_df.shape}")
    print(f"AIID subset shape: {aiid_df.shape}")

    # --- Analyze ATLAS (Theoretical/Research) ---
    # Identify the active tactics column
    tactic_cols = ['tactics', 'tactics_used']
    active_tactic_col = None
    for col in tactic_cols:
        if col in atlas_df.columns and atlas_df[col].notna().sum() > 0:
            active_tactic_col = col
            break
    
    atlas_total = len(atlas_df)
    atlas_evasion_hits = 0
    
    if active_tactic_col:
        # Normalize and search
        tactics_series = atlas_df[active_tactic_col].fillna('').astype(str).str.lower()
        # Search for 'evasion' or specific IDs
        evasion_mask = tactics_series.str.contains('evasion', na=False) | \
                       tactics_series.str.contains('aml.ta0007', na=False) | \
                       tactics_series.str.contains('aml.ta0005', na=False)
        atlas_evasion_hits = evasion_mask.sum()
    else:
        print("Warning: Could not identify an active tactics column in ATLAS data.")

    atlas_prop = atlas_evasion_hits / atlas_total if atlas_total > 0 else 0
    
    print(f"\nATLAS (Theoretical):")
    print(f"  Total Cases: {atlas_total}")
    print(f"  Cases with 'Evasion' tactics: {atlas_evasion_hits}")
    print(f"  Proportion: {atlas_prop:.4f}")

    # --- Analyze AIID (Real-World) ---
    # Identify the active failure column
    failure_cols = ['Known AI Technical Failure', '85'] # 85 is the index from metadata if names are stripped
    active_failure_col = None
    
    # First try exact name match
    if 'Known AI Technical Failure' in aiid_df.columns:
        active_failure_col = 'Known AI Technical Failure'
    else:
        # Try fuzzy match
        cols = [c for c in aiid_df.columns if 'Technical Failure' in str(c)]
        if cols:
            active_failure_col = cols[0]
    
    aiid_total = len(aiid_df)
    aiid_robustness_hits = 0
    
    if active_failure_col:
        # Normalize and search
        failures_series = aiid_df[active_failure_col].fillna('').astype(str).str.lower()
        # Search for 'robustness' or 'reliability'
        robustness_mask = failures_series.str.contains('robustness', na=False) | \
                          failures_series.str.contains('reliability', na=False)
        aiid_robustness_hits = robustness_mask.sum()
    else:
        print("Warning: Could not identify 'Known AI Technical Failure' column in AIID data.")

    aiid_prop = aiid_robustness_hits / aiid_total if aiid_total > 0 else 0
    
    print(f"\nAIID (Real-World):")
    print(f"  Total Incidents: {aiid_total}")
    print(f"  Incidents with 'Robustness/Reliability' failures: {aiid_robustness_hits}")
    print(f"  Proportion: {aiid_prop:.4f}")

    # --- Statistical Test ---
    stat, pval = 0.0, 1.0
    if atlas_total > 0 and aiid_total > 0:
        counts = np.array([atlas_evasion_hits, aiid_robustness_hits])
        nobs = np.array([atlas_total, aiid_total])
        
        # Two-sided Z-test
        stat, pval = proportions_ztest(counts, nobs)
        
        print(f"\n--- Statistical Comparison (Z-test) ---")
        print(f"Z-score: {stat:.4f}")
        print(f"P-value: {pval:.4e}")
        
        interpretation = "Significant difference" if pval < 0.05 else "No significant difference"
        print(f"Result: {interpretation}")
    else:
        print("\nInsufficient data for Z-test.")

    # --- Visualization ---
    labels = ['ATLAS\n(Theoretical Evasion)', 'AIID\n(Real-World Robustness)']
    proportions = [atlas_prop, aiid_prop]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, proportions, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    
    # Add exact numbers on bars
    for bar, prop in zip(bars, proportions):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, 
                 f'{prop:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    title_text = f"The 'Theory-Reality' Gap: Evasion vs. Robustness\n(p={pval:.4e})"
    plt.title(title_text, fontsize=14)
    plt.ylabel('Prevalence (Proportion of Dataset)')
    plt.ylim(0, max(proportions) * 1.2 if max(proportions) > 0 else 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotate with N
    if max(proportions) > 0:
        plt.text(0, atlas_prop/2 if atlas_prop > 0 else 0, f"n={atlas_total}", ha='center', color='white', fontweight='bold')
        plt.text(1, aiid_prop/2 if aiid_prop > 0 else 0, f"n={aiid_total}", ha='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Found dataset at: astalabs_discovery_all_data.csv
ATLAS subset shape: (52, 196)
AIID subset shape: (1362, 196)

ATLAS (Theoretical):
  Total Cases: 52
  Cases with 'Evasion' tactics: 33
  Proportion: 0.6346

AIID (Real-World):
  Total Incidents: 1362
  Incidents with 'Robustness/Reliability' failures: 4
  Proportion: 0.0029

--- Statistical Comparison (Z-test) ---
Z-score: 28.0055
P-value: 1.3929e-172
Result: Significant difference


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot (Categorical Comparison).
*   **Purpose:** The plot compares the prevalence (proportion) of a specific attribute (Evasion/Robustness) across two different datasets or contexts: "ATLAS" and "AIID". It aims to visualize the significant disparity—the "gap"—between theoretical findings and real-world observations.

### 2. Axes
*   **X-axis:**
    *   **Title/Labels:** The axis represents two distinct categories:
        1.  **ATLAS** (labeled "Theoretical Evasion")
        2.  **AIID** (labeled "Real-World Robustness")
    *   **Nature:** Categorical / Nominal data.
*   **Y-axis:**
    *   **Title/Label:** "Prevalence (Proportion of Dataset)".
    *   **Units:** The values represent ratios or proportions (decimal format), which are also interpreted as percentages in the annotations.
    *   **Value Range:** The axis ticks range from **0.0 to 0.7**, with the visual space extending slightly up to approximately 0.75.

### 3. Data Trends
*   **Tallest Bar (ATLAS):** The blue bar representing "Theoretical Evasion" is dominant, reaching a height of roughly 0.635 on the y-axis. This indicates a high prevalence of evasion-related entries in the ATLAS dataset.
*   **Shortest Bar (AIID):** The orange bar representing "Real-World Robustness" is extremely short, barely visible above the baseline, indicating a negligible prevalence (0.003 or 0.3%) in the AIID dataset.
*   **Trend/Pattern:** There is a drastic and visually stark contrast between the two categories. The plot demonstrates that "Evasion" is a primary focus or occurrence in the theoretical dataset (ATLAS), while it is virtually non-existent or rarely reported in the real-world dataset (AIID).

### 4. Annotations and Legends
*   **Bar Annotations (Percentages):**
    *   Above the **ATLAS** bar: **"63.5%"** is written in bold, indicating the exact proportion.
    *   Above the **AIID** bar: **"0.3%"** is written in bold.
*   **Bar Annotations (Counts):**
    *   Inside the ATLAS bar: **"n=52"**, likely representing the raw count of entries or the sample size associated with that percentage.
    *   Inside/Near the AIID bar: There is an "n" value annotation, partially obscured by the small size of the bar, indicating the raw count for that category.
*   **Chart Title:** "The 'Theory-Reality' Gap: Evasion vs. Robustness". This title provides the narrative frame for the data: a comparison between theory and reality.
*   **Statistical Annotation:** The title includes **"(p=1.3929e-172)"**. This indicates a statistical test result (likely a Chi-square or proportion test).
*   **Gridlines:** Horizontal dashed gridlines appear at intervals of 0.1 (0.0, 0.1, 0.2...) to assist with reading the y-values.

### 5. Statistical Insights
*   **The "Gap" is Confirmed:** The plot statistically validates the "Theory-Reality Gap." While 63.5% of the theoretical dataset (ATLAS) relates to Evasion, only 0.3% of the real-world dataset (AIID) relates to Robustness failures. This suggests that while researchers focus heavily on evasion attacks in theory, these attacks are rarely reported in real-world AI incidents.
*   **Extreme Statistical Significance:** The p-value of **1.3929e-172** is extremely close to zero. This implies that the difference between the two proportions is statistically significant to a very high degree. It confirms that the discrepancy is not due to random chance or sampling error; the distributions of these attributes in the two datasets are fundamentally different.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
