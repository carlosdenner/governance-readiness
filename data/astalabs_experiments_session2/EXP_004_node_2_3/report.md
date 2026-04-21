# Experiment 4: node_2_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_2_3` |
| **ID in Run** | 4 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:11:59.812909+00:00 |
| **Runtime** | 218.9s |
| **Parent** | `node_1_0` |
| **Children** | `node_3_2`, `node_3_9`, `node_3_19` |
| **Creation Index** | 5 |

---

## Hypothesis

> The Technical Implementability Gap: Governance requirements related to
'Fairness' and 'Explainability' map to significantly fewer unique architectural
controls than requirements related to 'Security', suggesting they are harder to
operationalize technically.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7742 (Likely True) |
| **Posterior** | 0.4286 (Maybe False) |
| **Surprise** | -0.4147 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 4.0 |
| Maybe True | 26.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare the density of architectural controls across different Trust Readiness domains.

### Steps
- 1. Load `context_crosswalk_evidence.json` (or filter `step2_crosswalk_matrix` from CSV).
- 2. Group the data by `bundle` or `trust_sub` (e.g., Fairness, Security, Transparency).
- 3. For each group, count the number of unique `applicable_controls` (or `arch_controls`) mapped.
- 4. Perform a one-way ANOVA or Kruskal-Wallis test to check for significant differences in control counts across domains.
- 5. Perform post-hoc pairwise comparisons to specifically test Security vs Fairness.

### Deliverables
- Box plot of control counts by Trust domain; Statistical test results verifying the 'Implementability Gap'.

---

## Analysis

The experiment successfully loaded the `context_crosswalk_evidence.json` dataset
and generated the requested statistics and visualizations. Due to the high-level
bundle labels ('Trust Readiness'), the code correctly utilized keyword analysis
on competency statements to categorize records into 'Security' (n=4), 'Fairness'
(n=2), and 'Explainability' (n=2), though the majority remained in 'Other'
(n=34).

The results **do not support** the hypothesis that 'Fairness' and
'Explainability' map to significantly fewer architectural controls than
'Security'.

1.  **Statistical Significance**: The Kruskal-Wallis test showed no significant
difference across groups (p=0.23), and pairwise Mann-Whitney U tests for
Security vs. Fairness (p=0.72) and Security vs. Explainability (p=0.78) were
also non-significant.
2.  **Directionality**: While 'Fairness' had a slightly lower mean control count
(2.00) than 'Security' (2.25), 'Explainability' actually had a **higher** mean
(2.50), directly contradicting the hypothesis for that domain.

The analysis is limited by the very small sample sizes identified for the
specific domains.

---

## Review

The experiment was executed faithfully according to the plan. The programmer
correctly identified that the high-level 'bundle' field ('Trust Readiness') was
insufficient for the granular comparison required and implemented a text-based
keyword extraction strategy to categorize records into 'Security', 'Fairness',
and 'Explainability'. While the resulting sample sizes were very small (Security
n=4, Fairness n=2, Explainability n=2), this reflects the dataset's constraints
rather than an implementation failure. The statistical tests (Kruskal-Wallis and
Mann-Whitney U) were appropriate for the non-parametric nature of the data, and
the boxplot visualization was successfully generated.

---

## Code

```python
import json
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Define file path: trying parent directory first as per instructions
filename = 'context_crosswalk_evidence.json'
filepath = f'../{filename}'
if not os.path.exists(filepath):
    filepath = filename # Fallback to current directory

print(f"Attempting to load data from: {filepath}")

try:
    # Load JSON data
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Process data into DataFrame
    records = []
    for item in data:
        # Extract fields
        bundle = item.get('bundle', 'Unknown')
        controls = item.get('applicable_controls', [])
        req_id = item.get('req_id', 'Unknown')
        statement = item.get('competency_statement', '')
        
        # Count controls (handle None or non-list types safely)
        if isinstance(controls, list):
            count = len(controls)
        else:
            count = 0
            
        records.append({
            'bundle': bundle,
            'control_count': count,
            'req_id': req_id,
            'statement': statement
        })
    
    df = pd.DataFrame(records)
    print(f"Data loaded. Shape: {df.shape}")
    
    # Check unique bundles
    unique_bundles = df['bundle'].unique()
    print(f"Unique Bundles found: {unique_bundles}")
    
    # If bundle names are generic or just one group, try to derive specific domains
    # The hypothesis compares Security vs Fairness/Explainability
    if len(unique_bundles) <= 1 or 'Trust Readiness' in unique_bundles[0]:
        print("Refining domains based on statement content...")
        def refine_domain(text):
            text = text.lower()
            if 'security' in text or 'attack' in text or 'adversar' in text: return 'Security'
            if 'fairness' in text or 'bias' in text: return 'Fairness'
            if 'explain' in text or 'interpret' in text or 'transparen' in text: return 'Explainability'
            if 'privacy' in text: return 'Privacy'
            return 'Other'
        
        df['analysis_group'] = df['statement'].apply(refine_domain)
        # Filter out 'Other' if we want cleaner plots, or keep them
        # For now, keep them but print counts
    else:
        df['analysis_group'] = df['bundle']

    print("\nGroup counts for analysis:")
    print(df['analysis_group'].value_counts())
    
    # Statistical Analysis
    print("\n--- Statistical Analysis ---")
    groups = [df[df['analysis_group'] == g]['control_count'].values for g in df['analysis_group'].unique()]
    
    if len(groups) > 1:
        stat, p = stats.kruskal(*groups)
        print(f"Kruskal-Wallis Test across all groups: H={stat:.4f}, p-value={p:.4f}")
    
    # Pairwise Comparisons for Hypothesis
    # Identify specific groups dynamically
    domain_groups = df['analysis_group'].unique()
    sec_group = next((g for g in domain_groups if 'Security' in g), None)
    fair_group = next((g for g in domain_groups if 'Fairness' in g), None)
    exp_group = next((g for g in domain_groups if 'Explainability' in g), None)
    
    def run_mann_whitney(g1_name, g2_name):
        if g1_name and g2_name:
            d1 = df[df['analysis_group'] == g1_name]['control_count']
            d2 = df[df['analysis_group'] == g2_name]['control_count']
            u_stat, p_val = stats.mannwhitneyu(d1, d2)
            print(f"\nMann-Whitney U Test: {g1_name} vs {g2_name}")
            print(f"  U-statistic: {u_stat:.2f}, p-value: {p_val:.4f}")
            print(f"  Mean Controls: {g1_name}={d1.mean():.2f}, {g2_name}={d2.mean():.2f}")
    
    run_mann_whitney(sec_group, fair_group)
    run_mann_whitney(sec_group, exp_group)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    df.boxplot(column='control_count', by='analysis_group', rot=45, grid=True)
    plt.title('Architectural Control Density by Trust Domain')
    plt.suptitle('') # Suppress default pandas title
    plt.ylabel('Number of Mapped Controls')
    plt.xlabel('Trust Domain')
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: File {filepath} not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Attempting to load data from: context_crosswalk_evidence.json
Data loaded. Shape: (42, 4)
Unique Bundles found: <StringArray>
['Trust Readiness', 'Integration Readiness']
Length: 2, dtype: str
Refining domains based on statement content...

Group counts for analysis:
analysis_group
Other             34
Security           4
Explainability     2
Fairness           2
Name: count, dtype: int64

--- Statistical Analysis ---
Kruskal-Wallis Test across all groups: H=4.2731, p-value=0.2334

Mann-Whitney U Test: Security vs Fairness
  U-statistic: 5.00, p-value: 0.7237
  Mean Controls: Security=2.25, Fairness=2.00

Mann-Whitney U Test: Security vs Explainability
  U-statistic: 3.00, p-value: 0.7799
  Mean Controls: Security=2.25, Explainability=2.50


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a box-and-whisker plot).
*   **Purpose:** This plot visualizes the distribution of numerical data ("Number of Mapped Controls") across different categorical groups ("Trust Domains"). It displays the median, quartiles, and potential outliers to compare the central tendency and variability of each group.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Trust Domain"
    *   **Labels:** The categories presented are "Explainability," "Fairness," "Other," and "Security." The labels are rotated at a 45-degree angle for readability.
*   **Y-Axis:**
    *   **Title:** "Number of Mapped Controls"
    *   **Value Range:** The visible scale ranges from 1.00 to 3.00, with tick marks at 0.25 intervals.
    *   **Units:** Count (numerical quantity of controls).

### 3. Data Trends
*   **Explainability:**
    *   This category has the **highest median** value (green line), situated at 2.5.
    *   The interquartile range (the blue box) spans from 2.25 to 2.75.
    *   The whiskers indicate a full range of data from 2.0 to 3.0. The distribution appears relatively symmetric.
*   **Fairness:**
    *   This plot is collapsed into a single line at the value of 2.0.
    *   This indicates **zero variance** in the data provided for this category; essentially, all data points for "Fairness" are equal to 2.
*   **Other:**
    *   The median is at 2.0.
    *   The box spans from 1.0 to 2.0, meaning the bottom 25% to 50% of the data falls within this range.
    *   There is a long upper whisker extending to 3.0, indicating a spread towards higher values despite a lower median.
*   **Security:**
    *   The median is at 2.0.
    *   The box is very compressed, ranging only from 2.0 to 2.25.
    *   There is a visible **outlier** (represented by a circle) at the value of 3.0.

### 4. Annotations and Legends
*   **Title:** "Architectural Control Density by Trust Domain" is clearly displayed at the top.
*   **Grid:** A standard grid is overlaid on the plot to assist in reading specific Y-axis values.
*   **Box Components (Implicit Legend):**
    *   **Green Line:** Represents the median of the data.
    *   **Blue Box:** Represents the Interquartile Range (IQR), containing the middle 50% of the data.
    *   **Black Whiskers:** Extend to the minimum and maximum values (excluding outliers).
    *   **Circle:** Represents an outlier point.

### 5. Statistical Insights
*   **Variability differences:** "Explainability" and "Other" show the most variability in the number of mapped controls. In contrast, "Fairness" is completely static, and "Security" has very low variability aside from a single outlier case.
*   **Central Tendency:** "Explainability" is the only domain with a median higher than 2.0. The other three domains ("Fairness," "Other," and "Security") all share a median of 2.0.
*   **Interpretation:** Generally, the "Explainability" domain requires or maps to a higher density of architectural controls (averaging 2.5) compared to the other trust domains. "Fairness" appears to be rigidly defined with exactly 2 controls in this dataset.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
