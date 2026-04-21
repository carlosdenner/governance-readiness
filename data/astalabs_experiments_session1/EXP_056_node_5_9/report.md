# Experiment 56: node_5_9

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_9` |
| **ID in Run** | 56 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:17:01.695569+00:00 |
| **Runtime** | 137.6s |
| **Parent** | `node_4_15` |
| **Children** | None |
| **Creation Index** | 57 |

---

## Hypothesis

> Competency statements derived from 'OWASP' requirements have significantly
higher 'Confidence' scores than those derived from 'NIST' or 'EU' requirements,
reflecting stronger consensus on security specificities.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7581 (Likely True) |
| **Posterior** | 0.3802 (Maybe False) |
| **Surprise** | -0.4386 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 2.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 90.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Evaluate if source material type influences the confidence of derived competencies.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Extract the framework source from 'req_id' (e.g., starts with 'NIST', 'EU', 'OWASP').
- 3. Convert 'confidence' column to numeric (High=3, Medium=2, Low=1).
- 4. Perform a One-Way ANOVA to compare mean confidence scores across the three sources.
- 5. Post-hoc Tukey test if significant.

### Deliverables
- Bar chart of Mean Confidence by Source and ANOVA results.

---

## Analysis

The experiment tested the hypothesis that competency statements derived from
'OWASP' requirements have significantly higher confidence scores than those from
'NIST' or 'EU' sources. The analysis of 42 competency statements (9 EU, 23 NIST,
10 OWASP) revealed that OWASP-derived statements indeed had a perfect mean
confidence score of 3.00 (indicating all were rated 'High'), compared to 2.89
for EU and 2.78 for NIST. However, the One-Way ANOVA yielded an F-statistic of
1.38 and a p-value of 0.2625. Since the p-value is greater than 0.05, the
difference in means is not statistically significant. Therefore, the hypothesis
is not supported statistically. While OWASP statements appear to have
universally high confidence in this dataset, the variance within the NIST and EU
groups combined with the sample sizes was not sufficient to declare a
significant divergence between the framework sources.

---

## Review

The experiment successfully tested the hypothesis regarding the influence of
framework source on competency confidence. The analysis of 42 competency
statements (23 NIST, 9 EU, 10 OWASP) revealed that while OWASP-derived
statements achieved a perfect mean confidence score of 3.00 (indicating 100%
'High' confidence), compared to 2.89 for EU and 2.78 for NIST, the difference
was not statistically significant. The One-Way ANOVA yielded a p-value of 0.2625
(> 0.05), failing to reject the null hypothesis. Thus, while OWASP sources
appear to yield universally high-confidence competencies in this specific
dataset, there is insufficient statistical evidence to claim they are
significantly superior to NIST or EU sources generally.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# [debug] Load dataset to inspect columns and values
# df_debug = pd.read_csv('../step2_competency_statements.csv')
# print(df_debug.head())
# print(df_debug['req_id'].unique())
# print(df_debug['confidence'].unique())

# 1. Load the dataset
file_path = '../step2_competency_statements.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    # Fallback for local testing if needed, though instructions say one level up
    df = pd.read_csv('step2_competency_statements.csv')

# 2. Extract Framework Source
def extract_source(req_id):
    if isinstance(req_id, str):
        if req_id.startswith('NIST'):
            return 'NIST'
        elif req_id.startswith('EU'):
            return 'EU'
        elif req_id.startswith('OWASP'):
            return 'OWASP'
    return 'Other'

df['source'] = df['req_id'].apply(extract_source)

# Filter out 'Other' if any (though metadata suggests these are the main ones)
df = df[df['source'] != 'Other']

# 3. Convert Confidence to Numeric
confidence_map = {'High': 3, 'Medium': 2, 'Low': 1, 'high': 3, 'medium': 2, 'low': 1}
df['confidence_score'] = df['confidence'].map(confidence_map)

# Remove rows with NaN confidence if any
df = df.dropna(subset=['confidence_score'])

# 4. Descriptive Statistics
group_stats = df.groupby('source')['confidence_score'].agg(['count', 'mean', 'std', 'sem'])
print("=== Descriptive Statistics by Source ===")
print(group_stats)
print("\n")

# 5. Statistical Test (One-Way ANOVA)
# Extract groups
groups = [df[df['source'] == s]['confidence_score'].values for s in ['NIST', 'EU', 'OWASP']]

# Check if we have enough data in each group
if all(len(g) > 1 for g in groups):
    f_stat, p_value = stats.f_oneway(*groups)
    
    print("=== One-Way ANOVA Results ===")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: Statistically Significant (p < 0.05)")
        # Post-hoc test (Tukey HSD) if significant
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tukey = pairwise_tukeyhsd(endog=df['confidence_score'], groups=df['source'], alpha=0.05)
            print("\n=== Tukey HSD Post-hoc Test ===")
            print(tukey)
        except ImportError:
            print("statsmodels not installed, skipping Tukey HSD.")
    else:
        print("Result: Not Statistically Significant (p >= 0.05)")
else:
    print("Insufficient data in one or more groups to perform ANOVA.")

# 6. Visualization
plt.figure(figsize=(10, 6))
bars = plt.bar(group_stats.index, group_stats['mean'], 
               yerr=group_stats['sem'], capsize=10, 
               color=['#4c72b0', '#55a868', '#c44e52'], alpha=0.8)

plt.title('Mean Confidence Score of Competency Statements by Source Framework')
plt.xlabel('Framework Source')
plt.ylabel('Mean Confidence Score (1=Low, 3=High)')
plt.ylim(0, 3.5)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Descriptive Statistics by Source ===
        count      mean       std       sem
source                                     
EU          9  2.888889  0.333333  0.111111
NIST       23  2.782609  0.421741  0.087939
OWASP      10  3.000000  0.000000  0.000000


=== One-Way ANOVA Results ===
F-statistic: 1.3844
P-value: 0.2625
Result: Not Statistically Significant (p >= 0.05)


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot (with error bars).
*   **Purpose:** The plot compares the mean confidence scores of competency statements across three different source frameworks. It is designed to visualize both the average performance (height of the bar) and the variability or uncertainty (error bars) associated with each source.

### 2. Axes
*   **X-axis:**
    *   **Title:** Framework Source
    *   **Labels:** Categorical variables representing three organizations: **EU**, **NIST**, and **OWASP**.
*   **Y-axis:**
    *   **Title:** Mean Confidence Score (1=Low, 3=High)
    *   **Range:** The visual axis ranges from **0.0 to 3.5**.
    *   **Intervals:** Tick marks are placed at every 0.5 unit.

### 3. Data Trends
*   **Tallest Bar:** **OWASP** has the highest mean score.
*   **Shortest Bar:** **NIST** has the lowest mean score among the three.
*   **General Pattern:** The scores are relatively close to one another, all falling within the upper end of the scale (between 2.78 and 3.00). This indicates a generally high level of confidence across all three frameworks.

### 4. Annotations and Legends
*   **Value Labels:** The exact mean values are annotated above each bar:
    *   EU: **2.89**
    *   NIST: **2.78**
    *   OWASP: **3.00**
*   **Error Bars:** Black "I" shaped bars are present on top of the colored bars. These represent the variability (likely Standard Error or Confidence Interval) of the data.
    *   The error bars for **EU** and **NIST** show visible variability above and below the mean.
    *   The error bar for **OWASP** is flat/compressed against the top value, suggesting extremely low or zero variance (a ceiling effect).

### 5. Statistical Insights
*   **High Confidence Overall:** Given the scale is 1 to 3, all three frameworks exhibit very high confidence scores (all > 2.7). This suggests that competency statements derived from these sources are generally viewed as highly reliable or clear by the evaluators.
*   **OWASP Perfection:** The OWASP framework achieved a perfect mean score of **3.00**. The lack of vertical extension in the error bar suggests there was no variation in the data; essentially, every data point for OWASP was likely a maximum score of 3.
*   **Comparison of Variability:**
    *   **NIST** (2.78) and **EU** (2.89) show some disagreement or variation among the scores, as indicated by the length of their error bars.
    *   **EU** performs slightly better than NIST on average, but the overlap in range suggests the distinction between them is less severe than the distinction between them and OWASP.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
