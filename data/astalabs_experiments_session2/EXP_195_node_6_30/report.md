# Experiment 195: node_6_30

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_30` |
| **ID in Run** | 195 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:14:04.661392+00:00 |
| **Runtime** | 346.9s |
| **Parent** | `node_5_45` |
| **Children** | None |
| **Creation Index** | 196 |

---

## Hypothesis

> The 'Gatekeeper Effect': Systems that have undergone an agency-level Internal
Review are significantly more likely to have secured a formal Authority to
Operate (ATO), quantifying the role of internal governance as a critical path to
authorization.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.9945 (Definitely True) |
| **Surprise** | +0.0128 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
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

**Objective:** Determine if 'Internal Review' status serves as a predictor for having an 'Authority to Operate' (ATO).

### Steps
- 1. Filter for 'eo13960_scored'.
- 2. Map '50_internal_review' to a binary predictor (Yes vs No).
- 3. Map '40_has_ato' to a binary outcome (Yes vs No).
- 4. Calculate the conditional probability of having an ATO given an Internal Review.
- 5. Run a Chi-Square test to check for statistical significance.

### Deliverables
- Contingency table; Probability calculation (P(ATO | Review) vs P(ATO | No Review)); Chi-Square test results.

---

## Analysis

The experiment successfully tested the 'Gatekeeper Effect' hypothesis,
confirming that internal governance reviews are a significant predictor of
obtaining an Authority to Operate (ATO). After correcting the data mapping logic
to handle descriptive text fields, the study analyzed 1,035 valid records from
the EO 13960 dataset.

The results show a strong correlation: systems that underwent an Internal Review
had an ATO rate of 63.06% (507/804), nearly double the 33.77% (78/231) rate for
systems without such reviews. A Chi-Square test yielded a statistic of 61.47 and
a p-value of 4.49e-15, overwhelmingly rejecting the null hypothesis. This
confirms that the Internal Review process acts as a critical 'gatekeeper,'
significantly increasing the likelihood of formal authorization.

---

## Review

The experiment was successfully executed and robustly handled the data quality
issues identified in previous iterations. By implementing fuzzy matching to
categorize the verbose text descriptions in the '50_internal_review' and
'40_has_ato' columns, the analysis recovered a significant portion of the
dataset (n=1,035). The statistical analysis (Chi-Square test, p=4.49e-15) and
probability calculations (63.06% vs 33.77%) provide strong, statistically
significant evidence supporting the 'Gatekeeper Effect' hypothesis. The
deliverables (contingency table, probabilities, and bar chart) are complete and
directly address the research question.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load the dataset
filepath = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filepath, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Total EO 13960 records: {len(eo_data)}")

# --- MAPPING LOGIC ---

def map_internal_review(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    if s == '' or s == 'nan':
        return np.nan
        
    # Explicit negatives
    if 'no documentation' in s:
        return False
    
    # Positives (Documentation exists in some form or explicit Yes)
    # Fixed syntax error here: added 'x' before 'in'
    if any(x in s for x in ['limited', 'developed', 'published', 'partially', 'yes']):
        return True
        
    return np.nan

def map_ato(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    if s == '' or s == 'nan':
        return np.nan
        
    # Explicit negatives
    if s.startswith('no'):
        return False
        
    # Explicit positives
    if 'yes' in s or 'operated' in s:
        return True
        
    # Ambiguous cases treated as NaN
    return np.nan

# Apply mappings
eo_data['has_review'] = eo_data['50_internal_review'].apply(map_internal_review)
eo_data['has_ato'] = eo_data['40_has_ato'].apply(map_ato)

# Filter valid data
clean_data = eo_data.dropna(subset=['has_review', 'has_ato'])

print(f"\nRecords after robust mapping and cleaning: {len(clean_data)}")
print(f"Dropped {len(eo_data) - len(clean_data)} records.")

# --- ANALYSIS ---

if len(clean_data) == 0:
    print("Insufficient data for analysis.")
else:
    # Contingency Table
    contingency_table = pd.crosstab(clean_data['has_review'], clean_data['has_ato'])
    # Check shape to assign correct index/column names
    if contingency_table.shape == (2, 2):
        contingency_table.index = ['No Review', 'Has Review']
        contingency_table.columns = ['No ATO', 'Has ATO']
    
    print("\nContingency Table (Count):")
    print(contingency_table)

    # Calculate Probabilities
    # P(ATO | Review)
    review_yes = clean_data[clean_data['has_review'] == True]
    p_ato_given_review = review_yes['has_ato'].mean() if len(review_yes) > 0 else 0
    
    # P(ATO | No Review)
    review_no = clean_data[clean_data['has_review'] == False]
    p_ato_given_no_review = review_no['has_ato'].mean() if len(review_no) > 0 else 0
    
    print(f"\nProbability of ATO given Internal Review: {p_ato_given_review:.2%}")
    print(f"Probability of ATO given NO Internal Review: {p_ato_given_no_review:.2%}")
    
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Visualization
    labels = ['No Internal Review', 'Internal Review']
    probs = [p_ato_given_no_review, p_ato_given_review]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, probs, color=['#d9534f', '#5bc0de'])
    plt.ylabel('Probability of Having ATO')
    plt.title('The Gatekeeper Effect: Internal Review vs ATO Status')
    plt.ylim(0, 1.0)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.1%}',
                 ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Conclusion
    alpha = 0.05
    if p < alpha:
        print("\nResult: Statistically Significant. The hypothesis is supported.")
    else:
        print("\nResult: Not Statistically Significant. The hypothesis is not supported.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total EO 13960 records: 1757

Records after robust mapping and cleaning: 1035
Dropped 722 records.

Contingency Table (Count):
            No ATO  Has ATO
No Review      153       78
Has Review     297      507

Probability of ATO given Internal Review: 63.06%
Probability of ATO given NO Internal Review: 33.77%

Chi-Square Statistic: 61.4724
P-value: 4.4900e-15

Result: Statistically Significant. The hypothesis is supported.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot is designed to compare the probability of a specific outcome (having "ATO") between two distinct categorical groups: those with "No Internal Review" and those with an "Internal Review."

### 2. Axes
*   **X-Axis:**
    *   **Label/Categories:** The axis represents the review status, split into two categories: **"No Internal Review"** and **"Internal Review"**.
*   **Y-Axis:**
    *   **Label:** **"Probability of Having ATO"**.
    *   **Units:** The axis is scaled in decimal probabilities ranging from **0.0 to 1.0**, representing a 0% to 100% scale.
    *   **Range:** The visible range is from 0.0 to 1.0, with tick marks at intervals of 0.2.

### 3. Data Trends
*   **Tallest Bar:** The blue bar representing the **"Internal Review"** category is the tallest, indicating a higher probability of success.
*   **Shortest Bar:** The red bar representing the **"No Internal Review"** category is the shortest.
*   **Pattern:** There is a distinct positive trend associated with the internal review process. The presence of an internal review corresponds to a significantly higher likelihood of having ATO compared to the absence of such a review.

### 4. Annotations and Legends
*   **Title:** **"The Gatekeeper Effect: Internal Review vs ATO Status"**. This suggests the plot is investigating whether the internal review acts as a critical filter or "gatekeeper" for success.
*   **Data Labels:**
    *   The red bar is annotated with **"33.8%"**.
    *   The blue bar is annotated with **"63.1%"**.
    *   These annotations convert the decimal y-axis values into easy-to-read percentages directly above the bars.
*   **Color Coding:**
    *   **Red:** Used for "No Internal Review," often visually associated with a negative state or stop.
    *   **Light Blue:** Used for "Internal Review," differentiating the positive intervention group.

### 5. Statistical Insights
*   **Significant Increase:** Implementing an Internal Review nearly doubles the probability of having ATO status. The probability jumps from **33.8%** (without review) to **63.1%** (with review).
*   **Absolute Difference:** There is a **29.3 percentage point increase** in the likelihood of obtaining ATO when an internal review is conducted.
*   **Relative Impact:** The "Internal Review" group is approximately **1.87 times more likely** to have ATO status than the "No Internal Review" group.
*   **Conclusion:** The data strongly supports the "Gatekeeper Effect" hypothesis presented in the title. The Internal Review process appears to be a highly effective predictor (or prerequisite) for achieving ATO status, suggesting that skipping this step significantly hampers the chances of success.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
