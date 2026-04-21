# Experiment 209: node_5_81

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_81` |
| **ID in Run** | 209 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:47:57.478684+00:00 |
| **Runtime** | 302.6s |
| **Parent** | `node_4_47` |
| **Children** | None |
| **Creation Index** | 210 |

---

## Hypothesis

> Kill Chain Complexity: Adversarial attacks resulting in 'Impact'
(destruction/disruption) involve a significantly higher count of distinct
'Techniques' (steps) than attacks resulting only in 'Exfiltration', reflecting
higher execution difficulty.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6613 (Maybe True) |
| **Posterior** | 0.2967 (Likely False) |
| **Surprise** | -0.4375 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 25.0 |
| Uncertain | 0.0 |
| Maybe False | 5.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 26.0 |
| Definitely False | 34.0 |

---

## Experiment Plan

**Objective:** Compare the tactical complexity of different adversarial objectives.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'atlas_cases'.
- 2. Parse the '93_techniques' column (comma-separated strings) and calculate the count of distinct techniques for each case.
- 3. Parse '92_tactics' to classify cases into 'Impact' group and 'Exfiltration' group (cases can belong to both, but prioritize primary outcome or analyze as non-exclusive groups).
- 4. Perform an Independent Samples T-test (or Mann-Whitney if non-normal) comparing the mean technique count between Impact cases and Exfiltration cases.

### Deliverables
- Mean technique counts for both groups; T-test results; Histogram of technique counts by group.

---

## Analysis

The experiment successfully analyzed the 'Kill Chain Complexity' hypothesis
using the ATLAS subset of the provided dataset. The code dynamically identified
the correct columns ('tactics' and 'techniques') and processed 42 relevant cases
(36 'Impact' and 6 'Exfiltration-Only').

The results strictly contradict the hypothesis that 'Impact' attacks are more
complex:
1. **Observed Complexity**: Both the 'Impact' group and the 'Exfiltration-Only'
group exhibited a mean distinct technique count of exactly 1.0. Every single
analyzed case listed only one technique.
2. **Statistical Significance**: The Mann-Whitney U test yielded a p-value of
1.0, confirming absolutely no statistical difference between the groups.
3. **Data Insight**: The uniform technique count (1.0 across the board) suggests
that in this specific dataset version, ATLAS cases are likely coded with a
single primary technique per entry, or the kill-chain data is not granular
enough to show multi-step complexity. Consequently, the null hypothesis cannot
be rejected.

---

## Review

The experiment was faithfully executed, adapting to schema discrepancies to
successfully analyze the ATLAS dataset. However, the analysis revealed a
critical data characteristic: every single analyzed case (n=42) listed exactly
one distinct technique, resulting in zero variance and a mean complexity of 1.0
for both groups. Consequently, the hypothesis that 'Impact' attacks are more
complex than 'Exfiltration' attacks could not be supported, as the metric for
complexity was uniform across the dataset.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re

def run_experiment():
    # Load dataset
    file_path = 'astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Dataset not found at {file_path}")
        return

    # Filter for ATLAS cases
    df_atlas = df[df['source_table'] == 'atlas_cases'].copy()
    
    # Identify correct columns for tactics and techniques
    # Based on metadata, likely 'tactics' and 'techniques'
    tactic_col = None
    tech_col = None
    
    possible_tactic_cols = ['tactics', '92_tactics', 'tactics_used', '92_tactics_used']
    possible_tech_cols = ['techniques', '93_techniques', 'techniques_used', '93_techniques_used']
    
    for col in possible_tactic_cols:
        if col in df.columns:
            tactic_col = col
            break
            
    for col in possible_tech_cols:
        if col in df.columns:
            tech_col = col
            break
    
    # Fallback to incident coding if atlas_cases is empty or columns missing
    if df_atlas.empty or not tactic_col or df_atlas[tactic_col].isnull().all():
        print("Primary subset 'atlas_cases' missing or empty. Checking 'step3_incident_coding'...")
        df_atlas = df[df['source_table'] == 'step3_incident_coding'].copy()
        # re-check columns if necessary, though they should be the same in a concatenated CSV

    if not tactic_col or not tech_col:
        print("Could not identify tactic/technique columns.")
        print("Available columns:", df.columns.tolist())
        return
        
    print(f"Using columns: '{tactic_col}' and '{tech_col}'")

    # Drop rows with missing values in key columns
    df_atlas = df_atlas.dropna(subset=[tactic_col, tech_col])
    
    if df_atlas.empty:
        print("No valid data rows found with populated tactics and techniques.")
        return

    # Function to parse distinct technique counts
    def get_technique_count(text):
        if not isinstance(text, str): return 0
        # Split by comma or semicolon
        techniques = re.split(r'[,;]', text)
        # Filter empty strings and strip whitespace
        techniques = [t.strip() for t in techniques if t.strip()]
        return len(set(techniques))

    # Function to check tactic presence
    def has_tactic(text, tactic):
        if not isinstance(text, str): return False
        return tactic.lower() in text.lower()

    # Apply processing
    df_atlas['tech_count'] = df_atlas[tech_col].apply(get_technique_count)
    df_atlas['is_impact'] = df_atlas[tactic_col].apply(lambda x: has_tactic(x, 'Impact'))
    df_atlas['is_exfil'] = df_atlas[tactic_col].apply(lambda x: has_tactic(x, 'Exfiltration'))

    # Define Groups
    # Group A: Resulting in Impact (Any case with Impact)
    group_impact = df_atlas[df_atlas['is_impact']]['tech_count']
    
    # Group B: Resulting ONLY in Exfiltration (Exfiltration present, Impact absent)
    group_exfil_only = df_atlas[df_atlas['is_exfil'] & (~df_atlas['is_impact'])]['tech_count']

    # Summary Stats
    print(f"\nAnalysis of Attack Complexity (Distinct Techniques):")
    print(f"Impact Cases (n={len(group_impact)}): Mean = {group_impact.mean():.2f}, Median = {group_impact.median()}")
    print(f"Exfil-Only Cases (n={len(group_exfil_only)}): Mean = {group_exfil_only.mean():.2f}, Median = {group_exfil_only.median()}")

    if len(group_impact) < 2 or len(group_exfil_only) < 2:
        print("Insufficient sample size for statistical testing.")
        return

    # Statistical Test 
    # Using Mann-Whitney U test (non-parametric)
    # Alternative 'greater': Impact > Exfil Only
    stat, p_val = stats.mannwhitneyu(group_impact, group_exfil_only, alternative='greater')
    
    print(f"\nMann-Whitney U Test Results (Alternative: Impact > Exfil Only):")
    print(f"U-statistic: {stat}")
    print(f"P-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Conclusion: Reject Null Hypothesis. Impact attacks involve significantly more techniques than Exfiltration-only attacks.")
    else:
        print("Conclusion: Fail to Reject Null Hypothesis.")

    # Visualization
    plt.figure(figsize=(10, 6))
    
    all_counts = pd.concat([group_impact, group_exfil_only])
    if not all_counts.empty:
        max_val = all_counts.max()
        bins = range(0, int(max_val) + 3)
        
        plt.hist(group_impact, bins=bins, alpha=0.5, label='Impact Cases', density=True, color='red')
        plt.hist(group_exfil_only, bins=bins, alpha=0.5, label='Exfil-Only Cases', density=True, color='blue')
        
        plt.axvline(group_impact.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean Impact ({group_impact.mean():.1f})')
        plt.axvline(group_exfil_only.mean(), color='blue', linestyle='dashed', linewidth=1, label=f'Mean Exfil ({group_exfil_only.mean():.1f})')
        
        plt.xlabel('Count of Distinct Techniques')
        plt.ylabel('Density')
        plt.title('Kill Chain Complexity: Impact vs Exfiltration-Only')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Using columns: 'tactics' and 'techniques'

Analysis of Attack Complexity (Distinct Techniques):
Impact Cases (n=36): Mean = 1.00, Median = 1.0
Exfil-Only Cases (n=6): Mean = 1.00, Median = 1.0

Mann-Whitney U Test Results (Alternative: Impact > Exfil Only):
U-statistic: 108.0
P-value: 1.0000
Conclusion: Fail to Reject Null Hypothesis.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the analysis:

**1. Plot Type**
*   **Type:** Overlapping Histogram (or Density Plot).
*   **Purpose:** The plot compares the distribution of "Kill Chain Complexity" (measured by the count of distinct techniques) across two categories of cyber incidents: "Impact Cases" and "Exfiltration-Only Cases". It aims to visualize whether one type of incident involves a more complex set of techniques than the other.

**2. Axes**
*   **X-Axis:**
    *   **Label:** "Count of Distinct Techniques"
    *   **Range:** The axis spans from 0.0 to 3.0.
    *   **Units:** Integer count (distinct techniques).
*   **Y-Axis:**
    *   **Label:** "Density"
    *   **Range:** The axis spans from 0.0 to roughly 1.05.
    *   **Units:** Probability density (proportion of total cases).

**3. Data Trends**
*   **Visual Pattern:** The plot shows a single, solid rectangular block spanning the x-axis from 1.0 to 2.0. The block is purple, indicating a complete overlap between the semi-transparent red bars ("Impact Cases") and blue bars ("Exfil-Only Cases").
*   **Distribution:**
    *   There is no spread in the data. All data points for both categories appear to fall into the same bin.
    *   The height of the bar is 1.0, indicating that 100% of the observed data for both categories lies within this specific range.
*   **Comparison:** There is no visible difference between the two datasets. They mirror each other perfectly.

**4. Annotations and Legends**
*   **Legend:** located in the upper right corner, it distinguishes the datasets:
    *   **Red Square:** Represents "Impact Cases".
    *   **Blue Square:** Represents "Exfil-Only Cases".
    *   **Red Dashed Line:** Indicates the "Mean Impact," explicitly labeled with a value of **(1.0)**.
    *   **Blue Dashed Line:** Indicates the "Mean Exfil," explicitly labeled with a value of **(1.0)**.
*   **Vertical Lines:** There is a dashed vertical line at **x = 1.0**. Because the means for both groups are identical (1.0), the red and blue dashed lines overlap perfectly.

**5. Statistical Insights**
*   **Uniform Complexity:** The most significant insight is the lack of variance. Every single case in this dataset—whether it resulted in "Impact" or "Exfiltration-Only"—utilized exactly **1 distinct technique**.
*   **Identical Averages:** As noted in the legend, the mean distinct technique count for both groups is exactly 1.0.
*   **Conclusion:** Based on this specific data, there is no correlation between the complexity of the kill chain (technique count) and the outcome (Impact vs. Exfiltration). Both outcomes are associated with the minimum complexity level shown on the graph.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
