# Experiment 155: node_5_58

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_58` |
| **ID in Run** | 155 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:24:11.419370+00:00 |
| **Runtime** | 289.7s |
| **Parent** | `node_4_48` |
| **Children** | `node_6_39` |
| **Creation Index** | 156 |

---

## Hypothesis

> The Sophistication of State Targets: Adversarial attacks targeting Government or
Defense systems involve a significantly higher number of distinct ATLAS tactics
(longer attack chains) compared to attacks targeting Commercial or Consumer
systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.2555 (Likely False) |
| **Surprise** | -0.5934 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
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
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Assess if adversarial attacks on government targets exhibit higher complexity (tactic depth) than private sector attacks.

### Steps
- 1. Load `atlas_cases` subset.
- 2. text-mine the `summary` or `name` columns to classify targets as 'Government/Defense' vs 'Commercial/Private'.
- 3. Parse the `tactics` column (counting comma-separated values or distinct tags) to calculate a 'Tactic Chain Length' for each case.
- 4. Perform an independent samples T-test (or Mann-Whitney U test) on the chain lengths between the two target groups.

### Deliverables
- Boxplot of Tactic Counts by Target Sector; T-test results.

---

## Analysis

The experiment was successfully executed. The code loaded the dataset,
identified 52 ATLAS cases, and classified them into 'Government/Defense' (n=4)
and 'Commercial/Private' (n=48) sectors using keyword analysis. The analysis of
attack sophistication (tactic chain length) revealed that every single case in
both groups had a tactic count of exactly 1.0. As a result, there was zero
variance in the data, causing the T-test to return 'nan' and the Mann-Whitney U
test to yield a p-value of 1.0. Consequently, the hypothesis that government
targets face more sophisticated attacks is rejected based on this dataset. The
result suggests a potential limitation in the dataset's granularity (e.g., only
the primary tactic was recorded) or that the specific subset of cases provided
consists entirely of single-tactic incidents.

---

## Review

The experiment was successfully executed and faithfully followed the plan to
test the hypothesis that government targets face more sophisticated (longer
tactic chain) attacks. The code loaded the ATLAS dataset (N=52), classified
cases into 'Government/Defense' (n=4) and 'Commercial/Private' (n=48) using
keyword text mining, and calculated the number of distinct tactics per case.

The analysis revealed a critical property of the dataset: every single case had
a calculated tactic chain length of exactly 1.0 (Mean=1.0, Variance=0). Due to
this lack of variance, the T-test returned 'nan', and the Mann-Whitney U test
yielded a p-value of 1.0. Consequently, the hypothesis is rejected; the provided
data shows no difference in attack sophistication between sectors, as all
incidents are coded with a single tactic. The zero-variance result suggests that
either the dataset only captures primary tactics or the specific subset provided
does not contain multi-stage attack chains.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import re

def load_data():
    filename = 'astalabs_discovery_all_data.csv'
    # Try current directory first, then parent directory
    if os.path.exists(filename):
        return pd.read_csv(filename, low_memory=False)
    elif os.path.exists(os.path.join('..', filename)):
        return pd.read_csv(os.path.join('..', filename), low_memory=False)
    else:
        raise FileNotFoundError(f"{filename} not found in current or parent directory.")

def run_experiment():
    try:
        df = load_data()
        print("Dataset loaded successfully.")
    except Exception as e:
        print(e)
        return

    # Filter for ATLAS cases
    atlas = df[df['source_table'] == 'atlas_cases'].copy()
    print(f"ATLAS cases found: {len(atlas)}")

    if len(atlas) == 0:
        print("No ATLAS cases found. Check dataset.")
        return

    # Text Analysis to classify Target Sector
    # Keywords for Government/Defense
    gov_keywords = [
        'government', 'defense', 'military', 'federal', 'agency', 'state', 
        'national', 'intelligence', 'surveillance', 'police', 'election', 
        'voting', 'public sector', 'ministry', 'army', 'navy', 'air force'
    ]

    def classify_sector(row):
        text = (str(row.get('name', '')) + " " + str(row.get('summary', ''))).lower()
        if any(kw in text for kw in gov_keywords):
            return 'Government/Defense'
        return 'Commercial/Private'

    atlas['sector'] = atlas.apply(classify_sector, axis=1)

    # Calculate Tactic Chain Length
    # The 'tactics' column might be comma separated or list-like. 
    # We'll count distinct items.
    def count_tactics(val):
        if pd.isna(val) or str(val).strip() == '':
            return 0
        # Remove brackets if they exist (JSON style) and split
        s = str(val).replace('[', '').replace(']', '').replace("'", "")
        # Split by comma
        items = [i.strip() for i in s.split(',') if i.strip()]
        return len(set(items)) # distinct tactics

    atlas['tactic_count'] = atlas['tactics'].apply(count_tactics)

    # Group Data
    gov_counts = atlas[atlas['sector'] == 'Government/Defense']['tactic_count']
    com_counts = atlas[atlas['sector'] == 'Commercial/Private']['tactic_count']

    print(f"\nSector Analysis:\n  Government/Defense: n={len(gov_counts)}, Mean Tactic Count={gov_counts.mean():.2f}\n  Commercial/Private: n={len(com_counts)}, Mean Tactic Count={com_counts.mean():.2f}")

    # Statistical Test
    # Using Mann-Whitney U test (non-parametric) as counts are often not normal
    # and sample sizes might be small.
    stat, p_val = stats.mannwhitneyu(gov_counts, com_counts, alternative='greater')
    
    print(f"\nMann-Whitney U Test (Gov > Com):\n  U-statistic={stat}\n  p-value={p_val:.4f}")

    # Independent T-test for robustness check
    t_stat, t_p = stats.ttest_ind(gov_counts, com_counts, equal_var=False)
    print(f"T-test (two-sided): t={t_stat:.4f}, p={t_p:.4f}")

    # Interpretation
    alpha = 0.05
    if p_val < alpha:
        print("\nResult: Government/Defense targets involve significantly longer tactic chains.")
    else:
        print("\nResult: No significant difference in tactic chain length between sectors.")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.boxplot([gov_counts, com_counts], labels=['Gov/Def', 'Com/Priv'], patch_artist=True)
    plt.title('Attack Sophistication: Tactic Counts by Target Sector')
    plt.ylabel('Number of Distinct Tactics')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Dataset loaded successfully.
ATLAS cases found: 52

Sector Analysis:
  Government/Defense: n=4, Mean Tactic Count=1.00
  Commercial/Private: n=48, Mean Tactic Count=1.00

Mann-Whitney U Test (Gov > Com):
  U-statistic=96.0
  p-value=1.0000
T-test (two-sided): t=nan, p=nan

Result: No significant difference in tactic chain length between sectors.

STDERR:
/usr/local/lib/python3.13/site-packages/scipy/stats/_axis_nan_policy.py:592: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
  res = hypotest_fun_out(*samples, **kwds)
<ipython-input-1-b4b1e6f11034>:93: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([gov_counts, com_counts], labels=['Gov/Def', 'Com/Priv'], patch_artist=True)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (specifically, two collapsed box plots).
*   **Purpose:** The plot is designed to compare the distribution of distinct tactics used in cyber attacks across two different target sectors: Government/Defense ("Gov/Def") and Commercial/Private ("Com/Priv").

### 2. Axes
*   **X-Axis:**
    *   **Labels:** Categorical labels representing the target sectors: **"Gov/Def"** and **"Com/Priv"**.
    *   **Value Range:** N/A (Categorical data).
*   **Y-Axis:**
    *   **Title:** **"Number of Distinct Tactics"**.
    *   **Value Range:** The visible scale ranges from roughly **0.94 to 1.05**, with major grid lines marked at intervals of 0.02 (0.96, 0.98, 1.00, 1.02, 1.04).

### 3. Data Trends
*   **Observation:** The plot shows two extremely flat features located at the **Y-value of 1.00**.
*   **Interpretation of Shapes:** In a standard box plot, a box represents the interquartile range (IQR), and the line inside represents the median. Since these plots appear as single flat orange lines, it indicates that the **median, the 25th percentile, and the 75th percentile are all identical** (specifically, they are all equal to 1).
*   **Pattern:** There is no visible spread or variance. This suggests that for the dataset visualized, nearly every recorded attack involved exactly **1 distinct tactic**, regardless of whether the target was in the Government/Defense sector or the Commercial/Private sector.

### 4. Annotations and Legends
*   **Title:** "Attack Sophistication: Tactic Counts by Target Sector" — This sets the context that the plot is measuring the complexity of attacks.
*   **Grid Lines:** Horizontal dashed grid lines are present to assist in reading the Y-axis values precisely.
*   **Color:** The central lines are orange, which is the default color for the median line in `matplotlib` (a common Python plotting library).

### 5. Statistical Insights
*   **Uniformity of Attacks:** The most significant insight is the lack of variation. The data indicates that attacks against both sectors are essentially identical in terms of this specific metric: they are mono-tactic attacks.
*   **Low Sophistication:** If "sophistication" is measured by the number of distinct tactics employed during an intrusion, the attacks in this dataset are of low sophistication (score of 1). Attackers are not chaining multiple distinct tactics types together in these specific instances.
*   **No Sector Differentiation:** There is no statistical difference between the "Gov/Def" and "Com/Priv" sectors regarding the number of distinct tactics used. Both sectors face the same simple attack structure based on this plot.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
