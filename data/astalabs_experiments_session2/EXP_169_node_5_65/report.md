# Experiment 169: node_5_65

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_65` |
| **ID in Run** | 169 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:01:44.068965+00:00 |
| **Runtime** | 386.0s |
| **Parent** | `node_4_36` |
| **Children** | None |
| **Creation Index** | 170 |

---

## Hypothesis

> The Security-Rights Divergence: 'Civilian' agencies (e.g., Health, Education)
demonstrate significantly higher compliance with rights-preserving controls
(Notice + Opt-Out) compared to 'Defense/Security' agencies, which prioritize
operational opacity.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.7473 (Likely True) |
| **Surprise** | +0.0064 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

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
| Maybe True | 60.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare governance styles between Civilian and Defense/Security agencies.

### Steps
- 1. Filter for `eo13960_scored`.
- 2. Categorize `3_agency` into 'Security' (DOD, DHS, DOJ, State) and 'Civilian' (HHS, DOE, VA, etc.).
- 3. Create a composite 'Rights Score' (0-2) summing `59_ai_notice` and `67_opt_out` (mapped to 0/1).
- 4. Perform an Independent Samples T-test or Mann-Whitney U test on the Rights Score between the two agency types.

### Deliverables
- Box plot of Rights Scores by Agency Type; T-test/Mann-Whitney results.

---

## Analysis

The experiment successfully tested the 'Security-Rights Divergence' hypothesis
using the EO 13960 dataset. With the corrected mapping logic applied to the
'Notice' and 'Opt-Out' columns, the analysis found a statistically significant
difference between the groups. The Mann-Whitney U test yielded a p-value of
0.034, supporting the hypothesis that Civilian agencies (Mean Score = 0.094) are
slightly more likely to implement rights-preserving controls than
Security/Defense agencies (Mean Score = 0.062).

However, the visualization and summary statistics reveal a 'floor effect' that
is practically more significant than the divergence itself. The median score for
both groups is 0, and the box plot shows that positive scores (1 or 2) are
outliers. This indicates that while Civilian agencies are statistically
'better', the vast majority of AI systems across the entire U.S. federal
government—regardless of sector—lack basic rights-preserving controls like
public notice and opt-out mechanisms.

---

## Review

The experiment successfully tested the 'Security-Rights Divergence' hypothesis
using the EO 13960 dataset. With the corrected mapping logic applied to the
'Notice' and 'Opt-Out' columns, the analysis found a statistically significant
difference between the groups. The Mann-Whitney U test yielded a p-value of
0.034, supporting the hypothesis that Civilian agencies (Mean Score = 0.094) are
slightly more likely to implement rights-preserving controls than
Security/Defense agencies (Mean Score = 0.062).

However, the visualization and summary statistics reveal a 'floor effect' that
is practically more significant than the divergence itself. The median score for
both groups is 0, and the box plot shows that positive scores (1 or 2) are
outliers. This indicates that while Civilian agencies are statistically
'better', the vast majority of AI systems across the entire U.S. federal
government—regardless of sector—lack basic rights-preserving controls like
public notice and opt-out mechanisms.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        exit(1)

# Filter for EO 13960
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# Define Security vs Civilian Agencies
# Security agencies: DOD, DHS, DOJ, State
security_keywords = ['Defense', 'Homeland', 'Justice', 'State']

def classify_agency(agency_name):
    if pd.isna(agency_name):
        return 'Civilian'
    name = str(agency_name)
    if any(keyword in name for keyword in security_keywords):
        return 'Security'
    return 'Civilian'

eo_df['agency_type'] = eo_df['3_agency'].apply(classify_agency)

# Map Notice (59_ai_notice)
# Identify non-compliance indicators to map to 0; others to 1.
negative_notice_starts = [
    'None of the above',
    'N/A',
    'Agency CAIO has waived',
    'AI is not safety'
]

def map_notice(val):
    if pd.isna(val):
        return 0
    s = str(val).strip()
    # If it starts with any negative phrase, it's a 0
    for neg in negative_notice_starts:
        if s.startswith(neg):
            return 0
    # Otherwise, assuming it indicates a method of notice (Online, In-person, Email, etc.)
    return 1

eo_df['notice_score'] = eo_df['59_ai_notice'].apply(map_notice)

# Map Opt-out (67_opt_out)
# Strict mapping: Only 'Yes' is 1.
def map_opt_out(val):
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s.lower() == 'yes':
        return 1
    return 0

eo_df['opt_out_score'] = eo_df['67_opt_out'].apply(map_opt_out)

# Calculate Composite Rights Score (0-2)
eo_df['rights_score'] = eo_df['notice_score'] + eo_df['opt_out_score']

# Separate Groups
security_scores = eo_df[eo_df['agency_type'] == 'Security']['rights_score']
civilian_scores = eo_df[eo_df['agency_type'] == 'Civilian']['rights_score']

# Perform Statistical Test (Mann-Whitney U)
stat, p_val = mannwhitneyu(civilian_scores, security_scores, alternative='two-sided')

# Output Results
print("--- Experiment Results: Security-Rights Divergence ---")
print(f"Security/Defense Agencies (N={len(security_scores)}): Mean Rights Score = {security_scores.mean():.3f}")
print(f"Civilian Agencies (N={len(civilian_scores)}): Mean Rights Score = {civilian_scores.mean():.3f}")
print(f"Mann-Whitney U Test: Statistic={stat}, p-value={p_val:.5f}")

if p_val < 0.05:
    print("Conclusion: Significant difference found. The hypothesis is supported.")
else:
    print("Conclusion: No significant difference found. The hypothesis is rejected.")

# Visualization
data_to_plot = [civilian_scores, security_scores]
plt.figure(figsize=(8, 6))
plt.boxplot(data_to_plot)
plt.xticks([1, 2], ['Civilian', 'Security/Defense'])
plt.title('Rights-Preserving Controls (Notice + Opt-Out) by Agency Type')
plt.ylabel('Rights Score (0-2)')
plt.yticks([0, 1, 2])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Experiment Results: Security-Rights Divergence ---
Security/Defense Agencies (N=385): Mean Rights Score = 0.062
Civilian Agencies (N=1372): Mean Rights Score = 0.094
Mann-Whitney U Test: Statistic=271935.0, p-value=0.03411
Conclusion: Significant difference found. The hypothesis is supported.


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided plot, here are the detailed findings:

**1. Plot Type**
*   **Type:** This is a **Box Plot** (or Box-and-Whisker Plot).
*   **Purpose:** It is used to compare the distribution of "Rights Scores" between two different categories of agencies ("Civilian" and "Security/Defense"). It is designed to visualize the median, quartiles, and outliers of the data.

**2. Axes**
*   **Title:** The chart title is "Rights-Preserving Controls (Notice + Opt-Out) by Agency Type".
*   **Y-Axis:**
    *   **Label:** "Rights Score (0-2)".
    *   **Range:** The axis displays discrete integer values marked at 0, 1, and 2.
*   **X-Axis:**
    *   **Label:** The axis categorizes the data by agency type.
    *   **Categories:** "Civilian" and "Security/Defense".

**3. Data Trends**
*   **Median and Quartiles (The Orange Lines):** For both the "Civilian" and "Security/Defense" categories, there is a solid orange line at the **0** mark on the Y-axis. In a box plot, this line represents the median.
*   **Collapsed Boxes:** The fact that there are no visible "boxes" (rectangles) around the orange median lines suggests that the Interquartile Range (IQR)—the distance between the 25th and 75th percentiles—is zero. This indicates that the vast majority of the data points for both agency types have a score of exactly **0**.
*   **Outliers:** Both categories display circles at Y-values of **1** and **2**. In box plots, these circles represent outliers. This means that while the norm is a score of 0, there are a few individual agencies in both categories that achieved scores of 1 or 2.

**4. Annotations and Legends**
*   **Gridlines:** The plot features horizontal dashed gridlines at the values 0, 1, and 2 to assist in reading the discrete scores.
*   **Markers:** The hollow black circles represent outlier data points that fall outside the typical distribution (which is heavily concentrated at 0).

**5. Statistical Insights**
*   **Low Baseline Performance:** The primary insight is that the overwhelming majority of agencies, regardless of whether they are Civilian or Security/Defense, received a **Rights Score of 0**. This suggests that "Rights-Preserving Controls" (defined as Notice + Opt-Out) are rarely implemented across the board.
*   **No Disparity:** There appears to be no significant statistical difference between Civilian and Security/Defense agencies regarding this metric. Both exhibit identical medians (0) and similar outlier patterns.
*   **Exceptions to the Rule:** The presence of outliers at 1 and 2 indicates that it is possible for agencies to implement these controls, but these cases are rare exceptions rather than the standard practice.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
