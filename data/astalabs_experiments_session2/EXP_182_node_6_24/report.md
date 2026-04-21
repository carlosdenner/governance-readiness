# Experiment 182: node_6_24

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_24` |
| **ID in Run** | 182 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:38:53.500028+00:00 |
| **Runtime** | 327.1s |
| **Parent** | `node_5_36` |
| **Children** | `node_7_19` |
| **Creation Index** | 183 |

---

## Hypothesis

> The 'Awakening' Lag: Federal AI systems initiated after 2020 show significantly
higher compliance with 'Disparity Mitigation' controls compared to systems
initiated prior to 2020.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6935 (Maybe True) |
| **Posterior** | 0.2473 (Likely False) |
| **Surprise** | -0.5356 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 27.0 |
| Uncertain | 0.0 |
| Maybe False | 3.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 4.0 |
| Definitely False | 56.0 |

---

## Experiment Plan

**Objective:** Test if the global focus on algorithmic justice post-2020 translated into federal implementation.

### Steps
- 1. Filter `eo13960_scored`.
- 2. Parse `18_date_initiated` or `year` to extract the start year.
- 3. Split data into `Post_2020` (>=2021) and `Pre_2020` (<=2020).
- 4. Compare the mean of `62_disparity_mitigation` (binary) between groups using a T-test.

### Deliverables
- 1. Temporal trend analysis of bias mitigation compliance.
- 2. T-test results.

---

## Analysis

The experiment tested the 'Awakening Lag' hypothesis, which posited that Federal
AI systems initiated after 2020 would show significantly higher compliance with
'Disparity Mitigation' controls due to increased global focus on algorithmic
justice. Using the EO 13960 dataset, 1,219 systems were analyzed. A text-
analysis heuristic categorized the 'Disparity Mitigation' field into 'Present'
(descriptive text) or 'Absent' (N/A, None, or empty).

**Results:**
- **Direction:** Contrary to the hypothesis, compliance/documentation rates
actually *decreased* over time.
  - **Pre-2020 (<=2020):** 6.6% of systems had descriptive mitigation plans.
  - **Post-2020 (>=2021):** Only 4.4% of systems had descriptive mitigation
plans.
- **Significance:** The T-test (testing for an increase) yielded a T-statistic
of -1.46 and a p-value of 0.93. The negative T-statistic confirms the trend is
in the opposite direction of the hypothesis.

**Conclusion:**
The hypothesis is **not supported**. The data suggests a regression in the
documentation of disparity mitigation strategies for newer federal AI systems
compared to older ones. This could indicate that while the volume of AI adoption
has increased (N=887 post-2020 vs N=332 pre-2020), the depth of governance
documentation regarding bias has not kept pace, or that legacy systems have been
retroactively documented more thoroughly than new, rapidly deployed systems.

---

## Review

The experiment was successfully executed. The programmer correctly resolved the
file path issue and implemented a robust text-analysis heuristic to classify the
unstructured 'Disparity Mitigation' field, overcoming the previous data parsing
failure. The statistical analysis and visualization effectively tested the
hypothesis.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# --- Step 1: Load Dataset ---
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(f'../{filename}'):
    filepath = f'../{filename}'
else:
    filepath = filename # Fail gracefully (or with error) if not found

print(f"Loading dataset from: {filepath}")
df = pd.read_csv(filepath, low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 rows: {len(eo_data)}")

# --- Step 2: Parse Dates ---
def parse_year(date_str):
    if pd.isna(date_str):
        return np.nan
    try:
        dt = pd.to_datetime(date_str, errors='coerce')
        if pd.notnull(dt):
            return dt.year
    except:
        pass
    return np.nan

eo_data['initiation_year'] = eo_data['18_date_initiated'].apply(parse_year)
eo_data_clean = eo_data.dropna(subset=['initiation_year']).copy()
print(f"Rows with valid initiation year: {len(eo_data_clean)}")

# --- Step 3: Classify Disparity Mitigation (Text Analysis) ---
# Heuristic: If text describes a process, it's a 1. If it says 'N/A', 'None', or is empty, it's a 0.

def classify_mitigation(val):
    if pd.isna(val):
        return 0
    
    text = str(val).strip().lower()
    
    # Check for empty or very short strings
    if len(text) < 4:
        return 0
        
    # Check for explicit negatives at the start
    negative_prefixes = ('n/a', 'na ', 'no ', 'none', 'not applicable', 'unknown', 'tbd')
    if text.startswith(negative_prefixes):
        return 0
        
    # Also check if the entire string is just 'no' or 'none' (handled by starts with logic mostly, but 'no' < 4 chars handled above)
    
    # If we are here, it likely contains descriptive text of a mitigation
    return 1

eo_data_clean['mitigation_score'] = eo_data_clean['62_disparity_mitigation'].apply(classify_mitigation)

# Debug: Check distribution
print("\nMitigation Score Distribution:")
print(eo_data_clean['mitigation_score'].value_counts())

# Debug: Show examples of 1s and 0s to verify heuristic
print("\nExamples of '1' (Mitigation Present):")
print(eo_data_clean[eo_data_clean['mitigation_score']==1]['62_disparity_mitigation'].head(3).tolist())
print("\nExamples of '0' (No Mitigation/NA):")
print(eo_data_clean[eo_data_clean['mitigation_score']==0]['62_disparity_mitigation'].head(3).tolist())

# --- Step 4: Split Groups & Statistical Test ---
group_pre = eo_data_clean[eo_data_clean['initiation_year'] <= 2020]
group_post = eo_data_clean[eo_data_clean['initiation_year'] >= 2021]

score_pre = group_pre['mitigation_score']
score_post = group_post['mitigation_score']

print(f"\nGroup Pre-2020 (<= 2020): N={len(score_pre)}")
print(f"Group Post-2020 (>= 2021): N={len(score_post)}")

mean_pre = score_pre.mean()
mean_post = score_post.mean()

print(f"Mean Compliance Pre-2020: {mean_pre:.4f}")
print(f"Mean Compliance Post-2020: {mean_post:.4f}")

# T-test
t_stat, p_val = stats.ttest_ind(score_post, score_pre, equal_var=False, alternative='greater')
print(f"\nT-test results (Post > Pre):")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# --- Step 5: Visualization ---
plt.figure(figsize=(10, 6))
means = [mean_pre, mean_post]
labels = ['Pre-2020 (<=2020)', 'Post-2020 (>=2021)']

# Calculate Standard Error
se_pre = score_pre.sem()
se_post = score_post.sem()

bars = plt.bar(labels, means, yerr=[se_pre, se_post], capsize=10, 
               color=['#cccccc', '#2ca02c'], alpha=0.9, width=0.6)

plt.ylabel('Proportion with Disparity Mitigation Description')
plt.title('The "Awakening" Lag: Federal AI Bias Mitigation (Pre vs Post 2020)')
plt.ylim(0, 1.0)

# Add annotations
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, 
             f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
EO 13960 rows: 1757
Rows with valid initiation year: 1219

Mitigation Score Distribution:
mitigation_score
0    1158
1      61
Name: count, dtype: int64

Examples of '1' (Mitigation Present):
['The threshold for the biometric matching was tested extensively with a variety of face types for several months to establish a match threshold for the identification.', 'The threshold for the biometric matching was tested extensively with a variety of face types for several months to establish a match threshold for the identification.', 'The threshold for the biometric matching was tested extensively with a variety of face types for several months to establish a match threshold for the identification.']

Examples of '0' (No Mitigation/NA):
[nan, nan, nan]

Group Pre-2020 (<= 2020): N=332
Group Post-2020 (>= 2021): N=887
Mean Compliance Pre-2020: 0.0663
Mean Compliance Post-2020: 0.0440

T-test results (Post > Pre):
T-statistic: -1.4564
P-value: 0.9271


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Bar Plot (with error bars).
*   **Purpose:** The plot compares the proportion of federal AI-related documents or actions that include a description of disparity mitigation across two distinct time periods: before and after the year 2020.

**2. Axes**
*   **X-Axis:**
    *   **Labels:** Categorical time periods: "Pre-2020 (<=2020)" and "Post-2020 (>=2021)".
*   **Y-Axis:**
    *   **Title:** "Proportion with Disparity Mitigation Description".
    *   **Value Range:** The axis ranges from 0.0 to 1.0 (representing 0% to 100%).
    *   **Formatting:** Major tick marks are placed at 0.2 intervals (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) with horizontal grid lines for readability.

**3. Data Trends**
*   **Comparison:** There are two bars showing a comparison between the two time periods.
*   **Pre-2020 (<=2020):** This is the taller bar (colored grey), representing a proportion of roughly **0.066**.
*   **Post-2020 (>=2021):** This is the shorter bar (colored green), representing a proportion of roughly **0.044**.
*   **Trend:** The data indicates a **decrease** in the proportion of disparity mitigation descriptions in the period after 2020 compared to the period before 2020.

**4. Annotations and Legends**
*   **Title:** "The 'Awakening' Lag: Federal AI Bias Mitigation (Pre vs Post 2020)". This title frames the data, suggesting that despite a presumed increase in awareness ("Awakening") regarding AI bias, the actual mitigation efforts have "lagged" or decreased.
*   **Data Labels:** Bold text explicitly labels the value of each bar: **6.6%** for the Pre-2020 group and **4.4%** for the Post-2020 group.
*   **Error Bars:** Both bars feature error bars (likely representing standard error or 95% confidence intervals), indicating the variability or uncertainty of the estimates.

**5. Statistical Insights**
*   **Counter-Intuitive Decline:** The most significant insight is that the inclusion of disparity mitigation descriptions in federal AI contexts has dropped by **2.2 percentage points** (from 6.6% to 4.4%) after 2020. This is a relative decrease of approximately 33%.
*   **Low Overall Adoption:** Regardless of the time period, the absolute proportions are very low. In both periods, fewer than 7% of the analyzed instances included descriptions of disparity mitigation, indicating that this is not yet a standard or widespread practice in the dataset analyzed.
*   **The "Lag" Phenomenon:** The title implies an expectation that attention to AI bias would have increased after 2020 (perhaps due to social movements or increased academic focus). However, the statistical evidence presented here contradicts that expectation, showing a regression rather than progress in documenting mitigation strategies.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
