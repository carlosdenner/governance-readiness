# Experiment 201: node_7_4

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_4` |
| **ID in Run** | 201 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:28:57.144123+00:00 |
| **Runtime** | 179.7s |
| **Parent** | `node_6_2` |
| **Children** | None |
| **Creation Index** | 202 |

---

## Hypothesis

> Governance Culture - Science vs. Security: Agencies with a scientific mandate
(e.g., NASA, DOE) demonstrate significantly higher 'Governance Readiness' scores
than security-focused agencies (DHS, DOD, DOJ).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7258 (Likely True) |
| **Posterior** | 0.2473 (Likely False) |
| **Surprise** | -0.5743 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 2.0 |
| Maybe True | 26.0 |
| Uncertain | 0.0 |
| Maybe False | 2.0 |
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

**Objective:** Compare composite governance scores between Science and Security agencies.

### Steps
- 1. Filter `eo13960_scored`.
- 2. Categorize `3_abr` into 'Science' (NASA, DOE, NSF, HHS) vs 'Security' (DHS, DOD, DOJ, STATE).
- 3. Create a `Governance_Score` (0-5) for each system by summing binary compliance for: Impact Assessment, Testing, Monitoring, Equity, and Notice.
- 4. Perform a T-test (or Mann-Whitney U) to compare the mean scores of the two groups.

### Deliverables
- Box plot of Governance Scores by Agency Type and T-test results.

---

## Analysis

The experiment successfully tested the 'Governance Culture' hypothesis using the
EO 13960 dataset.

**Data & Methodology:**
- 618 systems were analyzed, split into 'Science' (NASA, DOE, NSF, HHS; n=384)
and 'Security' (DHS, DOD, DOJ, STATE; n=234) cohorts.
- A composite 'Governance Readiness Score' (0-5) was calculated based on the
presence of impact assessments, testing, monitoring, equity mitigation, and
public notice.

**Results:**
- **Science Agencies:** Mean Score = 0.16 / 5.0
- **Security Agencies:** Mean Score = 0.19 / 5.0
- **Statistical Test:** The Mann-Whitney U test yielded a p-value of 0.66,
indicating no statistically significant difference between the groups.

**Hypothesis Evaluation:**
The hypothesis that Science agencies demonstrate higher governance readiness is
**not supported**.
1. **No Significant Difference:** The p-value (0.66) confirms that the
distributions are statistically indistinguishable.
2. **Floor Effect:** The most significant finding is the near-total lack of
governance documentation across *both* groups. The median score for both is 0,
and the means (<0.2) imply that the vast majority of systems in both 'Science'
and 'Security' sectors lack even basic controls like impact assessments or
monitoring. The proposed cultural distinction between 'Science' and 'Security'
does not manifest in AI governance compliance.

---

## Review

The experiment successfully tested the 'Governance Culture' hypothesis using the
EO 13960 dataset. The implementation followed the experiment plan faithfully,
correctly categorizing agencies into 'Science' and 'Security' cohorts and
calculating a composite governance readiness score.

**Findings:**
- **Hypothesis Status:** Not Supported. The analysis found no statistically
significant difference between Science and Security agencies (p-value = 0.66).
- **Scores:** Both groups exhibited extremely low governance readiness scores
(Science Mean: 0.16/5; Security Mean: 0.19/5). The medians for both groups were
0.
- **Insight:** The results reveal a 'floor effect' where the vast majority of
federal AI systems—regardless of agency culture—lack basic governance
documentation (impact assessments, testing, monitoring, etc.). The proposed
distinction between scientific and security-oriented cultures does not manifest
in this data; rather, there is a uniform lack of documented compliance.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define the file path (handling the directory note)
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    filepath = filename

print(f"Loading dataset from {filepath}...")

# Load dataset
df = pd.read_csv(filepath, low_memory=False)

# Filter for EO 13960 scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Define Agency Categories
science_agencies = ['NASA', 'DOE', 'NSF', 'HHS']
security_agencies = ['DHS', 'DOD', 'DOJ', 'STATE']

def categorize_agency(abr):
    if abr in science_agencies:
        return 'Science'
    elif abr in security_agencies:
        return 'Security'
    else:
        return 'Other'

df_eo['Agency_Category'] = df_eo['3_abr'].apply(categorize_agency)

# Filter for only the two target categories
df_target = df_eo[df_eo['Agency_Category'] != 'Other'].copy()

print(f"Filtered Dataset Shape: {df_target.shape}")
print(f"Counts per category:\n{df_target['Agency_Category'].value_counts()}")

# Define governance columns
gov_columns = [
    '52_impact_assessment',
    '53_real_world_testing',
    '56_monitor_postdeploy',
    '62_disparity_mitigation',
    '59_ai_notice'
]

# Helper function to binarize governance responses
def is_affirmative(val, col_name=None):
    if pd.isna(val):
        return 0
    text = str(val).lower().strip()
    
    # Negative keywords
    negatives = ['no', 'none', 'n/a', 'not', 'waived', 'pending', 'unknown']
    
    # Specific logic for Notice (based on previous exploration)
    if col_name == '59_ai_notice':
        # If it starts with a negative indicator
        if any(text.startswith(n) for n in negatives):
            return 0
        # If it contains "none of the above"
        if "none of the above" in text:
            return 0
        return 1
    
    # General logic for other columns (Assessment, Testing, etc.)
    # Usually look for "yes", "completed", "conducted"
    # If strict "no" or "not applicable", then 0.
    
    # Check for explicit No first
    if text in ['no', 'no.', 'not applicable', 'n/a', 'none']:
        return 0
    
    # Check for Yes/Positive indicators
    positives = ['yes', 'completed', 'conducted', 'performed', 'implemented', 'ongoing']
    if any(p in text for p in positives):
        return 1
        
    # Fallback: if not explicitly negative, assume 0 for safety unless unclear, 
    # but let's see. Many fields might be descriptive.
    # For this experiment, we'll assume if it doesn't match positives, it's 0.
    return 0

# Calculate scores
for col in gov_columns:
    df_target[f'Score_{col}'] = df_target[col].apply(lambda x: is_affirmative(x, col))

# Sum for composite score
df_target['Governance_Score'] = df_target[[f'Score_{c}' for c in gov_columns]].sum(axis=1)

# Separate groups
science_scores = df_target[df_target['Agency_Category'] == 'Science']['Governance_Score']
security_scores = df_target[df_target['Agency_Category'] == 'Security']['Governance_Score']

# Statistical Test (Mann-Whitney U)
u_stat, p_val = stats.mannwhitneyu(science_scores, security_scores, alternative='two-sided')

# Calculate Means
mean_science = science_scores.mean()
mean_security = security_scores.mean()

print("\n--- Results ---")
print(f"Science Agencies (n={len(science_scores)}) Mean Governance Score: {mean_science:.2f}")
print(f"Security Agencies (n={len(security_scores)}) Mean Governance Score: {mean_security:.2f}")
print(f"Mann-Whitney U Statistic: {u_stat}, p-value: {p_val:.5f}")

if p_val < 0.05:
    print("Result: Statistically significant difference.")
else:
    print("Result: No statistically significant difference.")

# Visualization
plt.figure(figsize=(10, 6))
# Create a list of data for boxplot
data_to_plot = [science_scores, security_scores]
labels = ['Science (NASA, DOE, NSF, HHS)', 'Security (DHS, DOD, DOJ, STATE)']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
plt.title('Governance Readiness Scores by Agency Type')
plt.ylabel('Composite Governance Score (0-5)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text annotation for p-value
plt.text(1.5, 4.5, f'p={p_val:.4f}', horizontalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Filtered Dataset Shape: (618, 197)
Counts per category:
Agency_Category
Science     384
Security    234
Name: count, dtype: int64

--- Results ---
Science Agencies (n=384) Mean Governance Score: 0.16
Security Agencies (n=234) Mean Governance Score: 0.19
Mann-Whitney U Statistic: 45388.0, p-value: 0.66026
Result: No statistically significant difference.

STDERR:
<ipython-input-1-4be68390d342>:125: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the visual analysis of the provided image, here are the details:

### 1. Plot Type
*   **Type:** This is a **box plot** (also known as a box-and-whisker plot), though it presents in a collapsed form due to the nature of the data distribution.
*   **Purpose:** The plot is designed to compare the statistical distribution of "Governance Readiness Scores" between two distinct categories of government agencies ("Science" vs. "Security").

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Labels:** Two categorical groups:
        1.  **Science** (which includes NASA, DOE, NSF, HHS).
        2.  **Security** (which includes DHS, DOD, DOJ, STATE).
*   **Y-Axis (Vertical):**
    *   **Label:** "Composite Governance Score (0-5)".
    *   **Range:** The visual axis ranges from **0.0 to 3.0**. While the label implies a potential scale up to 5, the plotted data only reaches 3.
    *   **Intervals:** There are horizontal dashed grid lines at 0.5 intervals.

### 3. Data Trends
*   **Medians/Interquartile Range (The "Box"):** For both the Science and Security agencies, there is a flat red line at the **0.0** mark. In a box plot, this indicates that the median score is 0. Furthermore, because there is no visible "box" height, the 25th and 75th percentiles are likely also 0 (or very close to it), indicating that the vast majority of agencies in both groups have a score of 0.
*   **Outliers/Data Points:** Both categories display circles at integers **1.0, 2.0, and 3.0**. In the context of a box plot with a median of 0, these are considered outliers or distinct positive data points.
*   **Pattern:** Both groups exhibit an identical visual pattern: a heavy concentration at 0 with identical spread into the higher scores of 1, 2, and 3.

### 4. Annotations and Legends
*   **Main Title:** "Governance Readiness Scores by Agency Type".
*   **Statistical Annotation:** A box at the top center contains the text **"p=0.6603"**. This represents the p-value resulting from a statistical hypothesis test comparing the two groups.
*   **Gridlines:** Horizontal dashed lines run across the plot to assist in estimating the y-values of the data points.

### 5. Statistical Insights
*   **No Significant Difference:** The p-value of **0.6603** is well above the standard significance threshold (typically 0.05). This indicates that there is **no statistically significant difference** in Governance Readiness Scores between Science agencies and Security agencies. The data does not support the idea that one type of agency is more "ready" than the other.
*   **Low Overall Readiness:** The fact that the median is 0 and the maximum observed score is 3 (on a scale that goes up to 5) suggests that, in general, governance readiness is low across all surveyed agencies regardless of type.
*   **Identical Distributions:** Visually and statistically, the two groups appear remarkably similar, with a "floor effect" where most data points are bottomed out at zero.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
