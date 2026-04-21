# Experiment 50: node_6_1

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_1` |
| **ID in Run** | 50 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:07:04.771209+00:00 |
| **Runtime** | 175.8s |
| **Parent** | `node_5_1` |
| **Children** | `node_7_0` |
| **Creation Index** | 51 |

---

## Hypothesis

> Competency statements classified as 'Trust Readiness' are linguistically more
complex (higher word count) than 'Integration Readiness' statements, reflecting
the verbosity of policy language vs. engineering brevity.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7339 (Likely True) |
| **Posterior** | 0.1880 (Likely False) |
| **Surprise** | -0.6335 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
| Uncertain | 1.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Analyze the linguistic characteristics of the competency bundles.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Calculate the word count for each 'competency_statement'.
- 3. Group the word counts by 'bundle'.
- 4. Perform an Independent Samples T-test on the word counts.
- 5. Generate a histogram or density plot of word counts for each bundle.

### Deliverables
- Descriptive statistics (mean word count), T-test results, and distribution plot.

---

## Analysis

The experiment successfully tested the hypothesis that 'Trust Readiness'
(governance) competency statements are more verbose than 'Integration Readiness'
(engineering) statements. The analysis of 42 competency statements revealed
virtually identical linguistic complexity between the two bundles. Trust
Readiness statements had a mean word count of 26.63 (SD=3.83), while Integration
Readiness statements had a mean word count of 26.65 (SD=4.14). An Independent
Samples T-test (Welch's) yielded a t-statistic of -0.0167 and a p-value of
0.9868, indicating no statistically significant difference. The histogram
visualization confirms this finding, showing largely overlapping distributions
with similar central tendencies, though Integration Readiness displayed slightly
higher variance with a long tail of outliers. Consequently, the hypothesis is
rejected; in this dataset, policy-oriented governance requirements are not
inherently more verbose than technical architecture controls.

---

## Review

The experiment successfully tested the linguistic complexity hypothesis by
analyzing word counts of 42 competency statements from
'step2_competency_statements.csv'. The analysis revealed virtually identical
lengths between the two groups: 'Trust Readiness' (governance) statements had a
mean word count of 26.63 (SD=3.83), while 'Integration Readiness' (engineering)
statements had a mean of 26.65 (SD=4.14). An Independent Samples T-test
(Welch's) confirmed no statistically significant difference (t=-0.0167,
p=0.9868). The histogram visualization further illustrated the overlapping
distributions, showing that policy-oriented statements are not inherently more
verbose than technical controls in this dataset. Consequently, the hypothesis is
rejected.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# Define file path based on the note provided
file_path = '../step2_competency_statements.csv'

# Fallback to current directory if not found (handling potential environment inconsistencies)
if not os.path.exists(file_path):
    file_path = 'step2_competency_statements.csv'

try:
    # 1. Load the dataset
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # 2. Calculate word count for each 'competency_statement'
    # Using simple whitespace splitting
    df['word_count'] = df['competency_statement'].apply(lambda x: len(str(x).split()))

    # 3. Group by 'bundle'
    trust_data = df[df['bundle'] == 'Trust Readiness']['word_count']
    integration_data = df[df['bundle'] == 'Integration Readiness']['word_count']

    # 4. Descriptive Statistics
    trust_desc = trust_data.describe()
    integration_desc = integration_data.describe()

    print("\n=== Descriptive Statistics (Word Count) ===")
    print(f"Trust Readiness (n={int(trust_desc['count'])})")
    print(f"  Mean: {trust_desc['mean']:.2f}")
    print(f"  Median: {trust_desc['50%']:.2f}")
    print(f"  Std Dev: {trust_desc['std']:.2f}")
    
    print(f"\nIntegration Readiness (n={int(integration_desc['count'])})")
    print(f"  Mean: {integration_desc['mean']:.2f}")
    print(f"  Median: {integration_desc['50%']:.2f}")
    print(f"  Std Dev: {integration_desc['std']:.2f}")

    # 5. Independent Samples T-test (Welch's t-test for unequal variances/sample sizes)
    t_stat, p_val = stats.ttest_ind(trust_data, integration_data, equal_var=False)

    print("\n=== Independent Samples T-test (Welch's) ===")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Conclusion: Reject null hypothesis. There is a significant difference in word counts.")
    else:
        print("Conclusion: Fail to reject null hypothesis. No significant difference in word counts.")

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    
    # Plotting histograms with density to compare shapes despite unequal sample sizes
    plt.hist(trust_data, bins=10, alpha=0.6, label='Trust Readiness', density=True, color='blue', edgecolor='black')
    plt.hist(integration_data, bins=10, alpha=0.6, label='Integration Readiness', density=True, color='orange', edgecolor='black')
    
    # Add vertical lines for means
    plt.axvline(trust_data.mean(), color='blue', linestyle='dashed', linewidth=1, label=f'Trust Mean ({trust_data.mean():.1f})')
    plt.axvline(integration_data.mean(), color='orange', linestyle='dashed', linewidth=1, label=f'Integration Mean ({integration_data.mean():.1f})')

    plt.title('Word Count Distribution: Trust vs. Integration Competencies')
    plt.xlabel('Word Count')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_competency_statements.csv

=== Descriptive Statistics (Word Count) ===
Trust Readiness (n=19)
  Mean: 26.63
  Median: 25.00
  Std Dev: 3.83

Integration Readiness (n=23)
  Mean: 26.65
  Median: 27.00
  Std Dev: 4.14

=== Independent Samples T-test (Welch's) ===
T-statistic: -0.0167
P-value: 0.9868
Conclusion: Fail to reject null hypothesis. No significant difference in word counts.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Overlapping Density Histogram.
*   **Purpose:** This plot compares the frequency distributions of word counts for two distinct categories: "Trust Readiness" and "Integration Readiness." By using density on the Y-axis rather than raw count, it allows for a direct comparison of the distribution shapes even if the total number of samples in each category differs.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Word Count"
    *   **Range:** The axis ticks range from roughly **21 to 39**. (Major ticks are marked at 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5).
*   **Y-Axis:**
    *   **Label:** "Density" (indicating probability density).
    *   **Range:** The values range from **0.000 to approximately 0.210**. (Major ticks are marked every 0.025 units).

### 3. Data Trends
*   **Trust Readiness (Blue Bars):**
    *   **Peak:** The highest density occurs in the bin centered around **25–26 words** (density ~0.200).
    *   **Secondary Peak:** There is a notable secondary cluster around **33–34 words**.
    *   **Spread:** The distribution is relatively contained between 21 and 34 words.
*   **Integration Readiness (Orange Bars):**
    *   **Peak:** The highest density is in the lower range, around **22–24 words** (density ~0.205).
    *   **Spread & Outliers:** This distribution is more spread out than the Trust dataset. It includes a noticeable outlier (or small cluster) on the far right, between **37.5 and 39 words**.
    *   **Shape:** It appears slightly right-skewed, with a heavy concentration on the lower end but a long tail extending to the right.

### 4. Annotations and Legends
*   **Legend (Top Right):**
    *   **Blue Square:** Represents the "Trust Readiness" histogram data.
    *   **Orange Square:** Represents the "Integration Readiness" histogram data.
    *   **Blue Dashed Line:** Indicates the mean word count for Trust Readiness, calculated at **26.6**.
    *   **Orange Dashed Line:** Indicates the mean word count for Integration Readiness, calculated at **26.7**.
*   **Vertical Lines:** Two dashed vertical lines appear on the plot corresponding to the means. Because the values (26.6 and 26.7) are nearly identical, the lines appear almost as a single, overlapping line.

### 5. Statistical Insights
*   **Identical Averages:** Despite the differences in the shape of the distributions, the mean word counts for both categories are virtually identical (26.6 vs. 26.7). This indicates that, on average, the length of text describing both competencies is the same.
*   **Variance in Distribution:**
    *   **Integration Readiness** shows higher variance. It has the shortest descriptions (high density at 22-24) but also the longest descriptions (the bar near 38).
    *   **Trust Readiness** appears somewhat bimodal, with descriptions clustering heavily in the mid-20s and another distinct group in the low-30s, but lacking the extreme length seen in the Integration data.
*   **Implication:** While the "center of gravity" for both topics is the same, "Integration Readiness" descriptions are more likely to be brief, balanced out by a few unusually long descriptions, whereas "Trust Readiness" descriptions are more consistently clustered around specific mid-range lengths.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
