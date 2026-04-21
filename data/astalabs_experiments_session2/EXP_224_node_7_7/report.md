# Experiment 224: node_7_7

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_7` |
| **ID in Run** | 224 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:27:10.951530+00:00 |
| **Runtime** | 258.0s |
| **Parent** | `node_6_18` |
| **Children** | None |
| **Creation Index** | 225 |

---

## Hypothesis

> The 'Paper Compliance' Hypothesis: Agencies that perform 'Impact Assessments'
are significantly more likely to also perform 'Real-World Testing' compared to
those that do not, validating that administrative assessment drives technical
verification.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.2661 (Likely False) |
| **Posterior** | 0.7500 (Likely True) |
| **Surprise** | +0.5806 |
| **Surprise Interpretation** | Strong Positive (hypothesis strengthened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 1.0 |
| Maybe False | 29.0 |
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

**Objective:** Determine if governance paperwork (assessments) correlates with actual technical rigor (testing) in EO13960 use cases.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for the 'eo13960_scored' subset.
- 2. Clean column '52_impact_assessment' into a binary variable (Assessment_Done: Yes/No), treating 'N/A' as NaN.
- 3. Clean column '53_real_world_testing' into a binary variable (Testing_Done: Yes/No).
- 4. Create a contingency table of Assessment vs. Testing.
- 5. Perform a Chi-square test of independence to see if Assessment_Done predicts Testing_Done.

### Deliverables
- Contingency table and Chi-square test statistics (p-value, degrees of freedom).

---

## Analysis

The experiment successfully tested the 'Paper Compliance' hypothesis using
improved data cleaning logic. By correctly mapping descriptive text fields to
binary categories, the sample size increased from 22 to 148 valid use cases. The
results show a statistically significant positive association (Fisher's Exact
p=0.0023) between administrative governance and technical rigor. Agencies that
completed an 'Impact Assessment' were significantly more likely to also perform
'Real-World Testing' (93.3%) compared to those that did not (73.9%). The Odds
Ratio of 4.95 indicates that the presence of an impact assessment is a strong
predictor of technical testing, supporting the hypothesis that governance
paperwork in this context is correlated with actual technical verification.

---

## Review

The experiment successfully tested the 'Paper Compliance' hypothesis using
improved data cleaning logic. By correctly mapping descriptive text fields to
binary categories, the sample size increased from 22 to 148 valid use cases. The
results show a statistically significant positive association (Fisher's Exact
p=0.0023) between administrative governance and technical rigor. Agencies that
completed an 'Impact Assessment' were significantly more likely to also perform
'Real-World Testing' (93.3%) compared to those that did not (73.9%). The Odds
Ratio of 4.95 indicates that the presence of an impact assessment is a strong
predictor of technical testing, supporting the hypothesis that governance
paperwork in this context is correlated with actual technical verification.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys

print("Starting 'Paper Compliance' Hypothesis Experiment (Attempt 2)...")

# 1. Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# 2. Filter for 'eo13960_scored'
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset shape: {subset.shape}")

# 3. Define Cleaning Logic

def clean_impact_assessment(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ['yes', 'true', '1']:
        return 'Yes'
    elif s in ['no', 'false', '0', 'planned or in-progress.']:
        return 'No'
    return np.nan

def clean_real_world_testing(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    
    # Check for specific keywords based on dataset inspection
    if s == 'yes':
        return 'Yes'
    if 'operational environment' in s and 'not been tested' not in s:
        # Covers "Performance evaluation..." and "Impact evaluation..."
        return 'Yes'
    if 'no testing' in s or 'waived' in s or 'benchmark evaluation' in s:
        # Benchmark explicitly says "not been tested in an operational environment"
        return 'No'
    if 'no' == s:
        return 'No'
    
    return np.nan

# 4. Apply Cleaning
subset['Assessment_Clean'] = subset['52_impact_assessment'].apply(clean_impact_assessment)
subset['Testing_Clean'] = subset['53_real_world_testing'].apply(clean_real_world_testing)

# 5. Drop NaNs
analysis_df = subset.dropna(subset=['Assessment_Clean', 'Testing_Clean'])
print(f"Rows with valid data: {len(analysis_df)}")

# 6. Generate Contingency Table
contingency = pd.crosstab(analysis_df['Assessment_Clean'], analysis_df['Testing_Clean'])
print("\nContingency Table (Count):")
print(contingency)

# Row percentages
row_pct = pd.crosstab(analysis_df['Assessment_Clean'], analysis_df['Testing_Clean'], normalize='index') * 100
print("\nContingency Table (Row %):")
print(row_pct.round(2))

# 7. Statistical Test
if contingency.size == 0:
    print("Empty contingency table.")
    sys.exit(0)

chi2, p, dof, expected = stats.chi2_contingency(contingency)
print("\n--- Chi-Square Test Results ---")
print(f"Chi2: {chi2:.4f}, p-value: {p:.4e}, dof: {dof}")

# Fisher's Exact Test if 2x2
if contingency.shape == (2, 2):
    oddsratio, p_fisher = stats.fisher_exact(contingency)
    print(f"Fisher's Exact Test p-value: {p_fisher:.4e}")
    print(f"Odds Ratio: {oddsratio:.4f}")
    final_p = p_fisher
else:
    final_p = p

if final_p < 0.05:
    print("Result: Significant association found.")
else:
    print("Result: No significant association found.")

# 8. Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
plt.title('Assessment vs Testing (Count)')

plt.subplot(1, 2, 2)
sns.heatmap(row_pct, annot=True, fmt='.1f', cmap='Greens')
plt.title('Assessment vs Testing (Row %)')
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting 'Paper Compliance' Hypothesis Experiment (Attempt 2)...
Subset shape: (1757, 196)
Rows with valid data: 148

Contingency Table (Count):
Testing_Clean     No  Yes
Assessment_Clean         
No                23   65
Yes                4   56

Contingency Table (Row %):
Testing_Clean        No    Yes
Assessment_Clean              
No                26.14  73.86
Yes                6.67  93.33

--- Chi-Square Test Results ---
Chi2: 7.8086, p-value: 5.1997e-03, dof: 1
Fisher's Exact Test p-value: 2.3424e-03
Odds Ratio: 4.9538
Result: Significant association found.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** The image displays two side-by-side **Heatmaps** representing contingency tables (cross-tabulations).
*   **Purpose:** The plots visualize the relationship between two categorical variables: `Assessment_Clean` and `Testing_Clean`.
    *   The **left plot** displays the absolute frequency (**Count**) of observations for each combination.
    *   The **right plot** displays the **Row Percentage**, showing the proportion of `Testing_Clean` categories within each `Assessment_Clean` group.

### 2. Axes
*   **X-Axis (Both Plots):**
    *   **Label:** `Testing_Clean`
    *   **Categories:** "No", "Yes"
*   **Y-Axis (Both Plots):**
    *   **Label:** `Assessment_Clean`
    *   **Categories:** "No", "Yes"
*   **Color Scale (Z-Axis equivalent):**
    *   **Left Plot (Count):** A blue gradient scale ranging from approximately **0 to 65**. Darker blue indicates a higher count.
    *   **Right Plot (Row %):** A green gradient scale ranging from approximately **10 to 95**. Darker green indicates a higher percentage.

### 3. Data Trends
**Left Plot: Assessment vs Testing (Count)**
*   **High Values:** The highest concentration of data is in the `Testing_Clean = Yes` column. The specific intersection of `Assessment_Clean = No` and `Testing_Clean = Yes` has the highest count (65), followed closely by `Assessment_Clean = Yes` and `Testing_Clean = Yes` (56).
*   **Low Values:** The lowest count is found at the intersection of `Assessment_Clean = Yes` and `Testing_Clean = No` (only 4 observations).
*   **Overall Volume:** The dataset is heavily skewed toward `Testing_Clean = Yes`.

**Right Plot: Assessment vs Testing (Row %)**
*   **Row 1 (Assessment_Clean = No):** Of the individuals with no assessment, **73.9%** had testing, while **26.1%** did not.
*   **Row 2 (Assessment_Clean = Yes):** Of the individuals with an assessment, a significantly higher proportion (**93.3%**) had testing, leaving only **6.7%** without testing.
*   **Pattern:** The heat intensity (dark green) shifts entirely to the right column (`Testing_Clean = Yes`), indicating that "Yes" is the dominant outcome for Testing regardless of the Assessment status.

### 4. Annotations and Legends
*   **Cell Annotations:** Each cell contains the exact numerical value represented by the color:
    *   Left Plot: Integer counts (23, 65, 4, 56).
    *   Right Plot: Percentages formatted to one decimal place (26.1, 73.9, 6.7, 93.3).
*   **Color Bars:**
    *   The **Blue Color Bar** (Left) acts as a legend for frequency magnitude.
    *   The **Green Color Bar** (Right) acts as a legend for percentage magnitude.

### 5. Statistical Insights
*   **Strong Association with Testing:** There is a very high prevalence of `Testing_Clean = Yes` across the entire dataset. In total, 121 out of 148 observed cases (approx. 82%) involve Testing=Yes.
*   **Correlation between Variables:** There appears to be a positive association between having an Assessment and having Testing.
    *   When an Assessment is present (`Yes`), the likelihood of having Testing increases from **73.9%** (baseline for No Assessment) to **93.3%**.
*   **Rare Phenotype:** The condition of having an Assessment (`Yes`) but *not* having Testing (`No`) is an outlier/rare event, occurring in only 4 cases (6.7% of the Assessment group).
*   **Data Imbalance:** The data is unbalanced. The `Assessment_Clean` groups are roughly split (88 "No" vs 60 "Yes"), but the `Testing_Clean` groups are heavily imbalanced toward "Yes".
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
