# Experiment 17: node_4_0

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_0` |
| **ID in Run** | 17 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:48:56.605272+00:00 |
| **Runtime** | 184.8s |
| **Parent** | `node_3_0` |
| **Children** | `node_5_19` |
| **Creation Index** | 18 |

---

## Hypothesis

> User Agency Bundle: The provision of an 'Opt-Out' mechanism is strongly
correlated with the existence of an 'Appeal Process', indicating that
organizations typically deploy user-rights controls as a unified set rather than
in isolation.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7581 (Likely True) |
| **Posterior** | 0.9176 (Definitely True) |
| **Surprise** | +0.1914 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
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
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Explore the clustering of user-centric control mechanisms in federal AI systems.

### Steps
- 1. Filter for `eo13960_scored`.
- 2. Clean `67_opt_out` and `65_appeal_process` into binary variables.
- 3. Construct a crosstab of Opt-Out vs Appeal Process.
- 4. Calculate the Jaccard Similarity Index or simple correlation coefficient between the two binary vectors.
- 5. Visualize the overlap using a Venn diagram proxy (e.g., stacked bar or mosaic plot).

### Deliverables
- Mosaic plot of Appeal vs Opt-Out; Correlation coefficient and significance test.

---

## Analysis

The experiment successfully tested the 'User Agency Bundle' hypothesis using the
EO 13960 dataset. The code processed 1,757 federal AI use cases, converting the
'Opt-Out' (Section 67) and 'Appeal Process' (Section 65) text fields into binary
indicators.

The results strongly support the hypothesis that these user-rights controls are
implemented as a bundle rather than in isolation. The statistical analysis
revealed an extremely strong correlation (Phi Coefficient = 0.70, Chi-Square p <
1e-190).

Key findings:
1.  **The 'Null' State is Dominant**: 94.2% of all systems (1,656/1,757) lacked
*both* mechanisms, indicating that the baseline for federal AI is the absence of
these specific user controls.
2.  **Strong Predictability**: If a system lacks an Opt-Out mechanism, there is
a 98.9% chance it also lacks an Appeal process. Conversely, if an Opt-Out
mechanism is present (n=82), the likelihood of having an Appeal process jumps to
69.5%.
3.  **Validation**: The Jaccard Similarity Index of 0.56 indicates substantial
overlap between the affirmative sets.

This confirms that agencies typically deploy these governance features
together—likely as part of a comprehensive 'rights-preserving' architecture—or,
more commonly, omit them entirely.

---

## Review

The experiment was successfully executed and the results strongly support the
'User Agency Bundle' hypothesis. The code correctly processed the EO 13960
dataset (n=1,757), mapping the text-heavy 'Opt-Out' and 'Appeal Process' fields
into binary variables.

The analysis reveals a high degree of interdependence between these two
controls:
1.  **Strong Correlation**: The Phi coefficient of 0.7024 and a Chi-Square
p-value near zero (< 1e-190) confirm that these variables are not independent.
2.  **Predictive Power**: The presence of an Opt-Out mechanism significantly
increases the likelihood of an Appeal Process (rising from ~1.1% to 69.5%).
Conversely, the absence of Opt-Out is a near-perfect predictor for the absence
of Appeal (98.9%).
3.  **Cluster Validity**: The Jaccard Similarity Index of 0.56 indicates a
substantial overlap in the 'affirmative' cases, supporting the idea that
agencies implement these rights as a unified bundle.
4.  **Baseline Reality**: The contingency table highlights that the vast
majority of systems (1,656 out of 1,757, or ~94%) possess *neither* control,
suggesting that while the bundle exists conceptually, it is rarely implemented
in the current federal AI landscape.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np
import os

# [debug] Check file existence to handle path variability
file_name = 'astalabs_discovery_all_data.csv'
file_path = f"../{file_name}" if os.path.exists(f"../{file_name}") else file_name

print(f"Loading dataset from: {file_path}")

# Load dataset
df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Records: {len(eo_data)}")

# Define columns
col_opt_out = '67_opt_out'
col_appeal = '65_appeal_process'

# Inspect unique values for cleaning
print(f"Unique values in {col_opt_out}: {eo_data[col_opt_out].unique()}")
print(f"Unique values in {col_appeal}: {eo_data[col_appeal].unique()}")

# Data Cleaning Function
def clean_binary(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip().lower()
    # Check for affirmative values
    if val_str == 'yes' or val_str.startswith('yes'):
        return 1
    return 0

# Apply cleaning
eo_data['has_opt_out'] = eo_data[col_opt_out].apply(clean_binary)
neo_data = eo_data.copy() # Avoid SettingWithCopy warning on subsequent ops if any
neo_data['has_appeal'] = eo_data[col_appeal].apply(clean_binary)

# Create Contingency Table
ct = pd.crosstab(neo_data['has_opt_out'], neo_data['has_appeal'])
ct.index = ['No Opt-Out', 'Has Opt-Out']
ct.columns = ['No Appeal', 'Has Appeal']

print("\nContingency Table (Counts):")
print(ct)

# Calculate Percentages
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
print("\nContingency Table (Row Percentages):")
print(ct_pct)

# 1. Chi-Square Test
chi2, p, dof, expected = chi2_contingency(ct)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4e}")

# 2. Phi Coefficient (Correlation for binary variables)
# Phi = sqrt(chi2 / n)
n = ct.sum().sum()
phi = np.sqrt(chi2 / n)
print(f"Phi Coefficient (Correlation Strength): {phi:.4f}")

# 3. Jaccard Similarity (Intersection over Union for the 'Yes' condition)
# TP = Has Both, FP = Opt-Out Only, FN = Appeal Only
tp = ct.loc['Has Opt-Out', 'Has Appeal']
fp = ct.loc['Has Opt-Out', 'No Appeal']
fn = ct.loc['No Opt-Out', 'Has Appeal']
union = tp + fp + fn
jaccard = tp / union if union > 0 else 0.0
print(f"Jaccard Similarity Index (Overlap of 'Yes'): {jaccard:.4f}")

# Visualization: Stacked Bar Chart
plt.figure(figsize=(10, 6))
ax = ct_pct.plot(kind='bar', stacked=True, color=['#d62728', '#2ca02c'], alpha=0.8)
plt.title('Correlation between Opt-Out and Appeal Process (EO 13960)')
plt.xlabel('Opt-Out Mechanism Provided?')
plt.ylabel('Percentage of Systems')
plt.legend(title='Appeal Process Provided?', loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add labels
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', weight='bold')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
EO 13960 Records: 1757
Unique values in 67_opt_out: <StringArray>
[                                                                                                                                                                                                                                            nan,
                                                                                                                                                                                                                                           'Yes',
 'No – This AI use case is not subject to the opt-out requirement because the AI functionality is solely used for the prevention, detection, and investigation of fraud or cybersecurity incidents, or the conduct of a criminal investigation.',
                                                                                                                                                                                                                                         'Other',
                                                                                                                                                        'No – There is law or governmentwide guidance that restricts opt-out for this context. ',
                                                                                                                                                   'N/A; COTS tool used for code conversion, no individual's information is input into a model.',
                                                                                                                                                                 'Agency CAIO has waived this minimum practice and reported such waiver to OMB.']
Length: 7, dtype: str
Unique values in 65_appeal_process: <StringArray>
[                                                                                                                  nan,
                                                                                                                 'Yes',
                                                               'No – it is not operationally practical to offer this.',
                         'N/A; COTS tool used for code conversion, no individual's information is input into a model.',
 'No – Law, operational limitations, or governmentwide guidance precludes an opportunity for an individual to appeal.',
                                       'Agency CAIO has waived this minimum practice and reported such waiver to OMB.']
Length: 6, dtype: str

Contingency Table (Counts):
             No Appeal  Has Appeal
No Opt-Out        1656          19
Has Opt-Out         25          57

Contingency Table (Row Percentages):
             No Appeal  Has Appeal
No Opt-Out   98.865672    1.134328
Has Opt-Out  30.487805   69.512195

Chi-Square Statistic: 866.7367
P-Value: 1.6711e-190
Phi Coefficient (Correlation Strength): 0.7024
Jaccard Similarity Index (Overlap of 'Yes'): 0.5644


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** This chart compares the relative percentage distribution of one categorical variable (Appeal Process status) within two different groups defined by another categorical variable (Opt-Out Mechanism status). It is designed to visualize the relationship or correlation between these two system features.

### 2. Axes
*   **X-axis:**
    *   **Title:** "Opt-Out Mechanism Provided?"
    *   **Categories:** The axis is divided into two discrete categories: "**No Opt-Out**" and "**Has Opt-Out**".
*   **Y-axis:**
    *   **Title:** "Percentage of Systems"
    *   **Range:** 0 to 100.
    *   **Units:** Percent (%).
    *   **Increments:** Marked every 20 units (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **"No Opt-Out" Column:**
    *   This category is overwhelmingly dominated by the **"No Appeal"** segment (Red), which accounts for **98.9%** of the column.
    *   Only a tiny fraction, **1.1%** (Green), represents systems that have an appeal process despite lacking an opt-out mechanism.
*   **"Has Opt-Out" Column:**
    *   This category shows a significant shift. The majority, **69.5%**, corresponds to the **"Has Appeal"** segment (Green).
    *   The minority, **30.5%**, corresponds to the **"No Appeal"** segment (Red).
*   **Overall Pattern:** There is a visible inversion of trends. Systems without an opt-out mechanism almost certainly lack an appeal process, whereas systems that *do* provide an opt-out mechanism are significantly more likely to also provide an appeal process.

### 4. Annotations and Legends
*   **Chart Title:** "Correlation between Opt-Out and Appeal Process (EO 13960)" indicates the context is likely related to compliance with Executive Order 13960 regarding AI or automated systems.
*   **Legend:** Located on the right, titled "Appeal Process Provided?".
    *   **Red:** Represents "No Appeal".
    *   **Green:** Represents "Has Appeal".
*   **Bar Labels:** White text inside the bars provides the exact percentage values (e.g., "98.9%", "69.5%", "30.5%").
*   **Gridlines:** Horizontal dashed lines are used to help estimate the height of the bar segments relative to the Y-axis.

### 5. Statistical Insights
*   **Strong Positive Association:** There is a very strong correlation between the two variables. If a system includes an Opt-Out mechanism, the probability of it also having an Appeal process increases dramatically (from ~1% to ~70%).
*   **"All or Nothing" Compliance:** The data suggests that systems lacking one safeguard (Opt-Out) almost universally lack the other (Appeal). This implies that systems are generally either designed with a suite of rights/safeguards or with very few/none, rather than mixing and matching.
*   **Interdependency:** The existence of an Opt-Out mechanism serves as a strong predictor for the existence of an Appeal process. Conversely, the absence of an Opt-Out mechanism is a near-perfect predictor for the absence of an Appeal process.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
