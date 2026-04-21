# Experiment 30: node_3_14

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_14` |
| **ID in Run** | 30 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:26:39.603396+00:00 |
| **Runtime** | 206.9s |
| **Parent** | `node_2_7` |
| **Children** | `node_4_16`, `node_4_34`, `node_4_47` |
| **Creation Index** | 31 |

---

## Hypothesis

> Privacy Oversight Evasion: 'Law Enforcement' AI deployments containing PII are
significantly more likely to bypass 'SAOP Review' (Senior Agency Official for
Privacy) compared to 'Benefits/Services' deployments, indicating a security-
exemption culture.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7661 (Likely True) |
| **Posterior** | 0.2830 (Likely False) |
| **Surprise** | -0.5798 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 3.0 |
| Maybe True | 27.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 8.0 |
| Definitely False | 52.0 |

---

## Experiment Plan

**Objective:** Detect patterns of governance avoidance in sensitive sectors.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Filter for rows where '29_contains_pii' is Affirmative.
- 3. Segment data into Topic Areas: 'Law Enforcement/Security' vs 'Benefits/Health/Services'.
- 4. Compare the rate of '30_saop_review' == 'NO' (or missing) between the two groups.
- 5. Run a Two-proportion Z-test.

### Deliverables
- Bar chart of SAOP Bypass rates by Topic; Z-test statistic.

---

## Analysis

The experiment successfully tested the 'Privacy Oversight Evasion' hypothesis
using the EO 13960 dataset. After filtering for systems containing PII, 117
relevant cases were identified (84 in Benefits/Services, 33 in Law
Enforcement/Security).

The results **strongly contradict the hypothesis**:
1.  **Observed Rates**: 'Benefits/Services' deployments exhibited a much higher
rate of SAOP review bypass (44.05%) compared to 'Law Enforcement/Security'
deployments (21.21%).
2.  **Statistical Test**: The one-sided Z-test (testing if LE > Benefits)
yielded a Z-statistic of -2.2946 and a p-value of 0.989. This high p-value
confirms we fail to reject the null hypothesis in the hypothesized direction.
3.  **Inversion**: The data suggests the inverse relationship is statistically
significant: non-security 'Benefits' agencies are approximately twice as likely
to bypass privacy oversight compared to Law Enforcement agencies in this
dataset.

---

## Review

The experiment successfully tested the 'Privacy Oversight Evasion' hypothesis
using the EO 13960 dataset. The code correctly filtered for systems containing
PII and segmented them into the target groups ('Law Enforcement/Security' vs.
'Benefits/Services'). The analysis revealed 117 relevant cases. Contrary to the
hypothesis, 'Benefits/Services' deployments exhibited a significantly higher
rate of SAOP review bypass (44.05%) compared to 'Law Enforcement/Security'
deployments (21.21%). The one-sided Z-test (testing for LE > Benefits) yielded a
p-value of 0.99, leading to a failure to reject the null hypothesis in the
predicted direction. The findings suggest the inverse: civilian service agencies
are more likely to bypass privacy oversight than security agencies in this
dataset.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

# Load dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored dataset
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# 1. Filter for rows where PII is present
pii_positive = ['Yes', 'yes', 'True', 'true', '1']
eo_pii = eo_df[eo_df['29_contains_pii'].astype(str).str.strip().isin(pii_positive)].copy()

print(f"Total records with PII: {len(eo_pii)}")

# 2. Segment into Groups
le_keywords = ['Law Enforcement', 'Justice', 'Security', 'Defense', 'Intelligence', 'Homeland']
svc_keywords = ['Health', 'Benefits', 'Services', 'Transportation', 'Education', 'Energy', 'Environment', 'Labor', 'Commerce', 'Housing', 'Agriculture']

def classify_topic(topic):
    topic_str = str(topic)
    if any(k in topic_str for k in le_keywords):
        return 'Law Enforcement/Security'
    elif any(k in topic_str for k in svc_keywords):
        return 'Benefits/Services'
    else:
        return 'Other'

eo_pii['group'] = eo_pii['8_topic_area'].apply(classify_topic)

# Filter only for the two groups of interest
analysis_df = eo_pii[eo_pii['group'].isin(['Law Enforcement/Security', 'Benefits/Services'])].copy()

print("\nCounts by Group:")
print(analysis_df['group'].value_counts())

# 3. Define Metric: Bypass Rate
# Bypass = 'No' or Missing (NaN)
def is_bypass(val):
    s = str(val).strip().lower()
    if s == 'no' or s == 'nan':
        return 1
    return 0

analysis_df['bypass_flag'] = analysis_df['30_saop_review'].apply(is_bypass)

# 4. Statistical Test
group_stats = analysis_df.groupby('group')['bypass_flag'].agg(['sum', 'count', 'mean'])
# Rename columns for clarity
group_stats.columns = ['bypassed', 'total', 'rate']

print("\nBypass Statistics:")
print(group_stats)

if len(group_stats) == 2:
    # Extract stats for z-test
    # Note: 'Benefits/Services' is likely at index 0, 'Law Enforcement/Security' at index 1 due to sorting
    le_stats = group_stats.loc['Law Enforcement/Security']
    ben_stats = group_stats.loc['Benefits/Services']
    
    print(f"\nLE Bypass Rate: {le_stats['rate']:.2%}")
    print(f"Benefits Bypass Rate: {ben_stats['rate']:.2%}")
    
    # Hypothesis: LE > Benefits
    # Alternative='larger' means prop(group1) > prop(group2)
    # We pass counts and nobs as lists: [count_LE, count_Ben], [nobs_LE, nobs_Ben]
    
    # FIXED: Use 'bypassed' instead of 'sum' since columns were renamed
    count = [le_stats['bypassed'], ben_stats['bypassed']]
    nobs = [le_stats['total'], ben_stats['total']]
    
    stat, pval = proportions_ztest(count, nobs, alternative='larger')
    
    print(f"\nZ-test Statistic (LE > Benefits): {stat:.4f}")
    print(f"P-value: {pval:.4e}")
    
    # Interpretation check
    if pval < 0.05:
        print("Result: Statistically Significant (Reject Null)")
    else:
        print("Result: Not Significant (Fail to Reject Null)")
        
    # 5. Visualization
    plt.figure(figsize=(8, 6))
    colors = ['skyblue', 'salmon']
    # Ensure order matches
    groups = ['Benefits/Services', 'Law Enforcement/Security']
    rates = [ben_stats['rate'], le_stats['rate']]
    
    plt.bar(groups, rates, color=colors, alpha=0.8)
    plt.ylabel('Rate of SAOP Review Bypass (No/Missing)')
    plt.title('Privacy Oversight Evasion: Law Enforcement vs Benefits')
    plt.ylim(0, 1.0)
    for i, v in enumerate(rates):
        plt.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient groups for comparison.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total records with PII: 171

Counts by Group:
group
Benefits/Services           84
Law Enforcement/Security    33
Name: count, dtype: int64

Bypass Statistics:
                          bypassed  total      rate
group                                              
Benefits/Services               37     84  0.440476
Law Enforcement/Security         7     33  0.212121

LE Bypass Rate: 21.21%
Benefits Bypass Rate: 44.05%

Z-test Statistic (LE > Benefits): -2.2946
P-value: 9.8912e-01
Result: Not Significant (Fail to Reject Null)


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot.
*   **Purpose:** The plot is designed to compare the rates of privacy oversight evasion (specifically, bypassing the SAOP review) across two distinct functional categories: "Benefits/Services" and "Law Enforcement/Security."

### 2. Axes
*   **Y-Axis:**
    *   **Title:** "Rate of SAOP Review Bypass (No/Missing)"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Ticks:** The axis is marked at 0.2 intervals (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
*   **X-Axis:**
    *   **Labels:** The axis represents two categories: "Benefits/Services" and "Law Enforcement/Security."

### 3. Data Trends
*   **Highest Value:** The "Benefits/Services" category (left bar, light blue) has the highest rate of review bypass.
*   **Lowest Value:** The "Law Enforcement/Security" category (right bar, salmon/light red) has the lowest rate.
*   **Comparison:** The bar representing "Benefits/Services" is visibly more than twice the height of the "Law Enforcement/Security" bar, indicating a significant disparity in how often these two groups bypass privacy reviews.

### 4. Annotations and Legends
*   **Plot Title:** "Privacy Oversight Evasion: Law Enforcement vs Benefits"
*   **Value Annotations:** Specific percentage values are placed directly above each bar for precision:
    *   **44.0%** above the Benefits/Services bar.
    *   **21.2%** above the Law Enforcement/Security bar.
*   **Color Coding:** The plot uses distinct colors to differentiate the categories (Blue for Benefits, Salmon/Red for Law Enforcement), though the labels on the x-axis serve as the primary identifier.

### 5. Statistical Insights
*   **Significant Discrepancy:** There is a 22.8 percentage point difference between the two categories.
*   **Relative Risk:** Systems or processes related to "Benefits/Services" are more than twice as likely (approx. 2.07 times) to bypass the SAOP (Senior Agency Official for Privacy) review compared to those in "Law Enforcement/Security."
*   **Implication:** This suggests that while Law Enforcement/Security sectors generally have a higher compliance rate (only ~1 in 5 bypass review), the Benefits/Services sector struggles significantly with oversight compliance, with nearly half (44%) of instances missing the required review.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
