# Experiment 251: node_8_2

| Property | Value |
|---|---|
| **Experiment ID** | `node_8_2` |
| **ID in Run** | 251 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:52:59.829933+00:00 |
| **Runtime** | 316.6s |
| **Parent** | `node_7_2` |
| **Children** | None |
| **Creation Index** | 252 |

---

## Hypothesis

> The Biometric Bias-Mitigation Gap: Federal AI systems involving 'Facial
Recognition' or 'Biometrics' report *lower* rates of specific 'Disparity
Mitigation' measures compared to the general inventory baseline, despite high
public concern.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8548 (Likely True) |
| **Posterior** | 0.2912 (Likely False) |
| **Surprise** | -0.6764 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 14.0 |
| Maybe True | 16.0 |
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

**Objective:** Evaluate if high-risk biometric systems are adequately mitigating bias compared to other systems.

### Steps
- 1. Filter for 'eo13960_scored'.
- 2. Create binary 'Is_Biometric' by searching '2_use_case_name' and '11_purpose_benefits' for keywords (face, facial, biometric, recognition, fingerprint).
- 3. Create binary 'Has_Mitigation' from '55_62_disparity_mitigation' (Yes vs No).
- 4. Perform a Chi-square test comparing Biometric vs Non-Biometric systems.
- 5. Calculate relative risk or proportions.

### Deliverables
- Contingency table, Chi-square stats, and comparison chart of mitigation rates.

---

## Analysis

The experiment successfully tested the 'Biometric Bias-Mitigation Gap'
hypothesis using the EO 13960 dataset.

1. **Data Processing**: The code successfully identified 190 'Biometric' systems
(using keywords like face, fingerprint, iris) and 1,567 'Non-Biometric' systems.
A semantic search was applied to the 'Disparity Mitigation' column to identify
affirmative mitigation measures (e.g., testing, auditing, human review).

2. **Statistical Results**: The Chi-square test yielded a statistic of 56.59
with a p-value of **0.0000** (<0.05), indicating a highly significant difference
between the groups.

3. **Hypothesis Evaluation**: The hypothesis (that biometric systems would
report *lower* mitigation rates) is **rejected**.
   - **Biometric Systems** showed a **10.53%** mitigation reporting rate
(20/190).
   - **Non-Biometric Systems** showed a **1.40%** mitigation reporting rate
(22/1567).

4. **Key Insight**: Contrary to the hypothesis, biometric systems are
significantly *more* likely (approx. 7.5x) to report disparity mitigation
measures than the general AI inventory. This reversal likely reflects the
intense regulatory and public scrutiny on facial recognition technology, forcing
agencies to adopt (or at least report) safeguards more frequently than for less
visible back-office AI tools. However, the absolute rate (~10%) remains low,
indicating that 90% of biometric systems still lack reported bias mitigation.

---

## Review

The experiment successfully tested the 'Biometric Bias-Mitigation Gap'
hypothesis using the EO 13960 dataset.

1. **Data Processing**: The pipeline correctly filtered the 'eo13960_scored'
subset and implemented robust keyword matching to classify 190 systems as
'Biometric' (using terms like 'face', 'iris', 'fingerprint') and 1,567 as 'Non-
Biometric'. Crucially, the programmer adapted the 'Has_Mitigation' logic to
handle free-text data in the '62_disparity_mitigation' column, using semantic
keywords (e.g., 'test', 'audit', 'bias') to identify affirmative controls rather
than relying on a non-existent binary flag.

2. **Statistical Results**: The Chi-square test yielded a statistic of 56.59
with a p-value of **0.0000**, indicating a highly statistically significant
association.

3. **Hypothesis Evaluation**: The original hypothesis—that biometric systems
would report *lower* mitigation rates due to negligence—was **rejected**.
   - **Biometric Systems** demonstrated a **10.53%** mitigation reporting rate
(20/190).
   - **Non-Biometric Systems** demonstrated a **1.40%** mitigation reporting
rate (22/1567).

4. **Key Insight**: Contrary to the hypothesis, biometric systems are
significantly *more* likely (approx. 7.5x) to report disparity mitigation
measures than the general federal AI inventory. This reversal likely reflects
intense regulatory and public scrutiny on facial recognition technology,
compelling agencies to document safeguards more rigorously than for less visible
back-office AI tools. However, the absolute rate (~10%) remains objectively low,
implying that 90% of sensitive biometric deployments still lack reported bias
mitigation.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Robust file loading
possible_paths = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    file_path = 'astalabs_discovery_all_data.csv' # Fallback

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 Scored dataset
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {eo_data.shape}")

# 1. Define Biometric Logic
biometric_keywords = ['face', 'facial', 'biometric', 'recognition', 'fingerprint', 'voice', 'iris', 'dna', 'palm', 'gait']

def is_biometric(row):
    text = str(row.get('2_use_case_name', '')) + " " + str(row.get('11_purpose_benefits', ''))
    text = text.lower()
    return any(keyword in text for keyword in biometric_keywords)

eo_data['Is_Biometric'] = eo_data.apply(is_biometric, axis=1)

# 2. Define Mitigation Logic (Semantic search)
mitigation_col = '62_disparity_mitigation'
positive_keywords = ['test', 'eval', 'monitor', 'assess', 'audit', 'review', 'mitigat', 'bias', 'fair', 'human', 'check', 'guardrail', 'feedback', 'retrain', 'update']

def has_mitigation(row):
    text = str(row.get(mitigation_col, ''))
    if text.lower() == 'nan':
        return False
    # Check for positive indicators
    text_lower = text.lower()
    if any(pk in text_lower for pk in positive_keywords):
        return True
    return False

eo_data['Has_Mitigation'] = eo_data.apply(has_mitigation, axis=1)

# 3. Create Contingency Table & Ensure Dimensions
contingency_table = pd.crosstab(eo_data['Is_Biometric'], eo_data['Has_Mitigation'])

# Force 2x2 shape
contingency_table = contingency_table.reindex(index=[False, True], columns=[False, True], fill_value=0)

# Rename for clarity
contingency_table.index = ['Non-Biometric', 'Biometric']
contingency_table.columns = ['No Mitigation', 'Has Mitigation']

print("\nContingency Table (Counts):")
print(contingency_table)

# 4. Calculate Stats
biometric_total = contingency_table.loc['Biometric'].sum()
non_biometric_total = contingency_table.loc['Non-Biometric'].sum()

biometric_rate = contingency_table.loc['Biometric', 'Has Mitigation'] / biometric_total if biometric_total > 0 else 0
non_biometric_rate = contingency_table.loc['Non-Biometric', 'Has Mitigation'] / non_biometric_total if non_biometric_total > 0 else 0

print(f"\nBiometric Systems Mitigation Rate: {biometric_rate:.2%} (N={biometric_total})")
print(f"Non-Biometric Systems Mitigation Rate: {non_biometric_rate:.2%} (N={non_biometric_total})")

# 5. Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4f}")

# 6. Visualization
labels = ['Biometric', 'Non-Biometric']
rates = [biometric_rate, non_biometric_rate]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, rates, color=['#d62728', '#1f77b4'], alpha=0.8)
plt.title('Disparity Mitigation Rates: Biometric vs. General AI Systems')
plt.ylabel('Proportion with Mitigation Measures')
plt.ylim(0, max(rates)*1.2 if max(rates) > 0 else 1.0)

# Add count labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    count = contingency_table.iloc[1-i, 1] # Biometric is index 1 (i=0), Non-Bio is index 0 (i=1)
    total = biometric_total if i == 0 else non_biometric_total
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{height:.1%}\n(n={count}/{total})', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
EO 13960 Scored subset shape: (1757, 196)

Contingency Table (Counts):
               No Mitigation  Has Mitigation
Non-Biometric           1545              22
Biometric                170              20

Biometric Systems Mitigation Rate: 10.53% (N=190)
Non-Biometric Systems Mitigation Rate: 1.40% (N=1567)

Chi-Square Statistic: 56.5895
P-Value: 0.0000


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** This plot compares the prevalence of disparity mitigation measures between two distinct categories of Artificial Intelligence systems: "Biometric" and "Non-Biometric" (General AI). It is designed to visualize a significant gap in implementation rates between these two groups.

### 2. Axes
*   **X-Axis:**
    *   **Label:** Represents the categories of AI systems.
    *   **Values:** Two categorical variables: "Biometric" and "Non-Biometric".
*   **Y-Axis:**
    *   **Label:** "Proportion with Mitigation Measures".
    *   **Units:** The axis is measured in decimal proportion (ranging from 0 to 1), representing percentages.
    *   **Range:** The axis is scaled from 0.00 to approximately 0.125, with major grid lines marked at intervals of 0.02 (0.00, 0.02, 0.04, ... 0.12).

### 3. Data Trends
*   **Tallest Bar:** The "Biometric" category (red bar) is the tallest, reaching a value slightly above 0.10.
*   **Shortest Bar:** The "Non-Biometric" category (blue bar) is significantly shorter, reaching a value just above 0.01.
*   **Comparison:** There is a stark contrast in height between the two bars. Visually, the Biometric bar is approximately 7 to 8 times taller than the Non-Biometric bar, indicating a much higher rate of mitigation implementation in biometric systems.

### 4. Annotations and Legends
*   **Bar Annotations:**
    *   **Biometric:** Annotated with **"10.5% (n=20/190)"**. This indicates that 10.5% of the systems in this category utilized mitigation measures. The sample size shows that 20 out of a total of 190 biometric systems were identified as having these measures.
    *   **Non-Biometric:** Annotated with **"1.4% (n=22/1567)"**. This indicates a much lower rate of 1.4%. However, the sample size is significantly larger, with 22 systems employing measures out of a massive total of 1,567.
*   **Grid Lines:** Horizontal dashed grid lines are included to assist in estimating the y-axis values for the bars.
*   **Title:** The chart is titled "Disparity Mitigation Rates: Biometric vs. General AI Systems."

### 5. Statistical Insights
*   **Disproportionate Implementation:** The data reveals a significant disparity in how mitigation measures are applied. Biometric systems are substantially more likely (10.5%) to include disparity mitigation measures compared to general AI systems (1.4%).
*   **Sample Size Context:** While the percentage for Non-Biometric systems is very low, the raw number of systems with mitigation (n=22) is actually slightly higher than the raw number for Biometric systems (n=20). This highlights that the low percentage in the general category is driven by the vast denominator (1,567 total systems) rather than a lack of absolute examples.
*   **Research Implication:** This suggests that the field of Biometrics may be under higher scrutiny regarding bias and disparity, leading to a higher frequency of reported mitigation efforts relative to the total number of systems developed. Conversely, general AI systems appear to lack widespread adoption of these specific mitigation measures.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
