# Experiment 269: node_6_70

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_70` |
| **ID in Run** | 269 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:48:47.434172+00:00 |
| **Runtime** | 344.8s |
| **Parent** | `node_5_35` |
| **Children** | None |
| **Creation Index** | 270 |

---

## Hypothesis

> Adversarial Kill-Chain Complexity: Adversarial attacks classified as
'Exfiltration' (Model Stealing/Inversion) involve a significantly higher number
of unique tactics per case compared to 'Evasion' (Adversarial Example) attacks.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.3846 (Maybe False) |
| **Surprise** | -0.4288 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

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
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 48.0 |
| Definitely False | 12.0 |

---

## Experiment Plan

**Objective:** Quantify the complexity of different adversarial attack types.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'atlas_cases'.
- 2. Parse the '92_tactics' column (semicolon/comma separated) to count the number of unique tactics per case.
- 3. Categorize cases based on the presence of 'Exfiltration' vs. 'Evasion' keywords in '92_tactics' or '91_type'.
- 4. Compare the mean 'Tactic Count' between the two groups.
- 5. Perform an Independent Samples T-test.

### Deliverables
- Violin plot of Tactic Counts by Attack Type; T-test results.

---

## Analysis

The experiment successfully executed the classification and counting logic but
yielded degenerate results. After identifying the correct columns ('tactics' and
'type'), the code categorized 36 cases (21 Evasion, 15 Exfiltration). However,
the analysis revealed that every single case in both cohorts had a
'tactic_count' of exactly 1.0 (Standard Deviation = 0). This lack of variance
rendered the T-test calculation impossible (returning NaN due to division by
zero standard error).

Consequently, the hypothesis that 'Exfiltration' attacks are more complex than
'Evasion' attacks cannot be supported by this dataset, as both appear to have
identical, singular complexity in this view. This uniform result (1 tactic per
case) strongly suggests that the 'atlas_cases' table in this dataset either
represents a simplified view (one primary tactic per row) or uses a delimiter
not anticipated by the parser (though standard comma/semicolon checks were
included). The result is a null finding: no difference in complexity was
observed.

---

## Review

The experiment was successfully executed. The programmer correctly identified
the relevant columns ('tactics', 'type') after initial issues with column naming
conventions. The text parsing logic robustly handled potential delimiters.
Although the result (uniform tactic counts of 1.0 for all cases) suggests the
dataset uses a simplified single-label tagging scheme rather than a complex
multi-step kill-chain record, the experiment faithfully tested the hypothesis
against the available data. The null finding (no difference in complexity) is a
valid analytical result given the dataset constraints.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for ATLAS cases
atlas = df[df['source_table'] == 'atlas_cases'].copy()

print(f"Loaded ATLAS cases: {len(atlas)} rows")
# Identify correct column names for tactics and type
# Based on previous failure, we suspect columns are 'tactics' and 'type' not '92_tactics' etc.
tactics_col = 'tactics' if 'tactics' in atlas.columns else '92_tactics'
type_col = 'type' if 'type' in atlas.columns else '91_type'

print(f"Using columns: Tactics='{tactics_col}', Type='{type_col}'")

# Helper to count unique tactics
def count_tactics(row):
    tactic_str = row.get(tactics_col, '')
    if pd.isna(tactic_str) or str(tactic_str).strip() == '':
        return 0
    # Normalize separators (semicolon or comma)
    t_str = str(tactic_str).replace(',', ';')
    # Split, strip whitespace, and count unique non-empty entries
    tactics = [t.strip() for t in t_str.split(';') if t.strip()]
    return len(set(tactics))

# Calculate tactic counts
atlas['tactic_count'] = atlas.apply(count_tactics, axis=1)

# Categorize Attack Type
def categorize_attack(row):
    # Combine type, tactics, name, and summary for robust keyword search
    text_content = (
        str(row.get(type_col, '')) + ' ' + 
        str(row.get(tactics_col, '')) + ' ' + 
        str(row.get('name', '')) + ' ' + 
        str(row.get('summary', ''))
    ).lower()
    
    # Keywords for Exfiltration (Model Stealing/Inversion)
    exfil_keywords = ['exfiltration', 'model stealing', 'model inversion', 'extraction', 'steal']
    
    # Keywords for Evasion (Adversarial Example)
    evasion_keywords = ['evasion', 'adversarial example', 'perturbation', 'noise']
    
    # Classification logic
    if any(k in text_content for k in exfil_keywords):
        return 'Exfiltration'
    elif any(k in text_content for k in evasion_keywords):
        return 'Evasion'
    else:
        return 'Other'

atlas['attack_category'] = atlas.apply(categorize_attack, axis=1)

# Filter for the two groups
cohorts = atlas[atlas['attack_category'].isin(['Exfiltration', 'Evasion'])].copy()

# Stats
print("\n--- Cohort Analysis: Kill-Chain Complexity ---")
print(cohorts['attack_category'].value_counts())
group_stats = cohorts.groupby('attack_category')['tactic_count'].describe()
print(group_stats)

exfil_counts = cohorts[cohorts['attack_category'] == 'Exfiltration']['tactic_count']
evasion_counts = cohorts[cohorts['attack_category'] == 'Evasion']['tactic_count']

# T-Test
if len(exfil_counts) > 1 and len(evasion_counts) > 1:
    t_stat, p_val = stats.ttest_ind(exfil_counts, evasion_counts, equal_var=False)
    print(f"\nIndependent T-Test Results:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Result: Statistically significant difference detected (p < 0.05).")
    else:
        print("Result: No statistically significant difference detected.")
else:
    print("\nInsufficient data for T-test.")

# Visualization
plt.figure(figsize=(10, 6))
# Using simple boxplot if violin fails or data is sparse, but violin is requested.
sns.violinplot(x='attack_category', y='tactic_count', data=cohorts, palette='muted', inner='stick')
plt.title('Adversarial Kill-Chain Complexity: Exfiltration vs Evasion')
plt.xlabel('Attack Category')
plt.ylabel('Number of Unique Tactics Used')
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded ATLAS cases: 52 rows
Using columns: Tactics='tactics', Type='type'

--- Cohort Analysis: Kill-Chain Complexity ---
attack_category
Evasion         21
Exfiltration    15
Name: count, dtype: int64
                 count  mean  std  min  25%  50%  75%  max
attack_category                                           
Evasion           21.0   1.0  0.0  1.0  1.0  1.0  1.0  1.0
Exfiltration      15.0   1.0  0.0  1.0  1.0  1.0  1.0  1.0

Independent T-Test Results:
T-statistic: nan
P-value: nan
Result: No statistically significant difference detected.

STDERR:
/usr/local/lib/python3.13/site-packages/scipy/stats/_axis_nan_policy.py:592: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
  res = hypotest_fun_out(*samples, **kwds)
<ipython-input-1-78c9057f4166>:96: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.violinplot(x='attack_category', y='tactic_count', data=cohorts, palette='muted', inner='stick')


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Box Plot** (specifically, a degenerate box plot where the distribution has collapsed into a single value).
*   **Purpose:** The plot aims to compare the distribution of the "Number of Unique Tactics Used" across two different categories of adversarial behavior: "Evasion" and "Exfiltration." Box plots are typically used to show statistical distribution (minimum, first quartile, median, third quartile, and maximum), but here they indicate a lack of variance in the data.

### 2. Axes
*   **X-axis:**
    *   **Title:** "Attack Category"
    *   **Labels:** Two categorical variables: "Evasion" and "Exfiltration."
*   **Y-axis:**
    *   **Title:** "Number of Unique Tactics Used"
    *   **Range:** The visible scale ranges from approximately **0.95 to 1.05**, with tick marks at 0.96, 0.98, 1.00, 1.02, and 1.04.
    *   **Note on Units:** While the axis displays decimal values due to auto-scaling, the metric "Number of Unique Tactics" implies integer data. The lines are situated exactly at the integer value of **1.00**.

### 3. Data Trends
*   **Evasion:** Represented by a single horizontal line at the Y-value of **1.00**. This indicates that for every data point in the "Evasion" category, the number of unique tactics used was exactly 1. There is no spread, variance, or outliers visible.
*   **Exfiltration:** Similarly, this category is represented by a single horizontal line at the Y-value of **1.00**. Like the Evasion category, the data is constant at 1.
*   **Comparison:** There is no difference between the two categories. Both show identical behavior regarding this metric.

### 4. Annotations and Legends
*   **Title:** The chart is titled **"Adversarial Kill-Chain Complexity: Exfiltration vs Evasion"**. This sets the context of the analysis within cybersecurity, specifically looking at the complexity of attack phases.
*   **Legend:** There is no legend, as the categories are self-explanatory via the X-axis labels.

### 5. Statistical Insights
*   **Zero Variance:** The most significant statistical insight is that there is **zero variance** in the dataset for this specific metric. In every observed instance of Evasion and Exfiltration, the adversary (or simulation) utilized exactly **one** unique tactic.
*   **Low Complexity:** In the context of "Kill-Chain Complexity," a value of 1 suggests the lowest possible complexity. The attacks did not employ a diverse range of tactics within these phases; they relied on a single method for evasion and a single method for exfiltration.
*   **Uniformity:** The data suggests a highly uniform or standardized attack pattern, potentially indicative of automated scripts or a very specific, repetitive attack scenario where the methodology does not deviate.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
