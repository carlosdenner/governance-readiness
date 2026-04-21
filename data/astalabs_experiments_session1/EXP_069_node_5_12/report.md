# Experiment 69: node_5_12

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_12` |
| **ID in Run** | 69 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:45:35.524948+00:00 |
| **Runtime** | 264.1s |
| **Parent** | `node_4_18` |
| **Children** | None |
| **Creation Index** | 70 |

---

## Hypothesis

> Incidents where 'Human-in-the-Loop' mechanisms are identified as a missing
control are significantly more likely to result in 'Reliability' or 'Safety'
harms compared to 'Security' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.3698 (Maybe False) |
| **Surprise** | -0.4319 |
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
| Maybe False | 87.0 |
| Definitely False | 3.0 |

---

## Experiment Plan

**Objective:** Evaluate the efficacy of human oversight across different harm domains.

### Steps
- 1. Load 'step3_enrichments.json'.
- 2. Create a binary flag 'Missing_Human_Control' by searching for substrings 'human', 'approval', or 'override' (case-insensitive) in the 'missing_controls' column.
- 3. Create a binary target variable: 'Safety_Reliability' (if harm_type in ['reliability', 'safety', 'physical_safety']) vs 'Security' (if harm_type == 'security').
- 4. Filter dataset to include only these harm types.
- 5. Perform a Chi-square test or Fisher's Exact test on the contingency table.

### Deliverables
- Contingency table, test statistic, p-value.

---

## Analysis

The experiment successfully loaded the 'step3_enrichments.json' dataset and
performed a Fisher's Exact Test to evaluate the association between missing
'Human-in-the-Loop' controls and harm types (Safety/Reliability vs. Security).

**Key Findings:**
- **Data Distribution:** The dataset contained 40 relevant incidents: 36
classified as 'Security' harms and only 4 as 'Safety' or 'Reliability' harms.
- **Descriptive Statistics:**
    - Among 'Safety/Reliability' incidents (n=4), **0** (0%) were identified as
having missing human controls.
    - Among 'Security' incidents (n=36), **3** (8.3%) were identified as having
missing human controls.
- **Statistical Significance:** The Fisher's Exact Test yielded an Odds Ratio of
0.0000 and a p-value of **1.0000**.

**Conclusion:**
The hypothesis that missing human oversight is significantly more likely to
result in 'Reliability' or 'Safety' harms is **not supported** by this dataset.
In fact, no safety incidents in this sample were explicitly coded with missing
human controls, though the very small sample size for safety incidents (n=4)
limits the statistical power of this conclusion. The identified gaps in human
oversight were exclusively found in a small subset of Security incidents.

---

## Review

The experiment was faithfully implemented and successfully executed. The code
loaded the dataset, correctly categorized the harm types and missing controls,
and performed the appropriate statistical test (Fisher's Exact Test) suitable
for the small sample size.

---

## Code

```python
import json
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Handle file path: Try current directory first, as previous attempt with '../' failed
file_name = 'step3_enrichments.json'
if os.path.exists(file_name):
    file_path = file_name
elif os.path.exists(os.path.join('..', file_name)):
    file_path = os.path.join('..', file_name)
else:
    # Fallback to absolute check or raise error
    raise FileNotFoundError(f"{file_name} not found in current or parent directory.")

print(f"Loading dataset from: {file_path}")

with open(file_path, 'r') as f:
    data = json.load(f)

# Prepare data for analysis
records = []
for entry in data:
    # specific fields needed
    missing_controls = str(entry.get('missing_controls', '')).lower()
    harm_type = entry.get('harm_type', '').lower()
    
    # 1. Determine if Human-in-the-Loop is a missing control
    # Keywords: 'human', 'approval', 'override'
    # Note: Checking for 'human' might be too broad (e.g. 'humanoid'), but given the domain 'human-in-the-loop' or 'human oversight' is likely.
    # 'approval' covers 'approval gates', 'override' covers 'human override'.
    has_human_missing = any(keyword in missing_controls for keyword in ['human', 'approval', 'override'])
    
    # 2. Categorize Harm Type
    if harm_type in ['reliability', 'safety', 'physical_safety']:
        harm_category = 'Safety_Reliability'
    elif harm_type == 'security':
        harm_category = 'Security'
    else:
        harm_category = None # Exclude other harm types (privacy, bias, etc.)
    
    if harm_category:
        records.append({
            'case_study_id': entry.get('case_study_id'),
            'missing_controls': missing_controls,
            'harm_type': harm_type,
            'Missing_Human_Control': has_human_missing,
            'Harm_Category': harm_category
        })

# Create DataFrame
df = pd.DataFrame(records)

if df.empty:
    print("No records matched the criteria.")
else:
    print("=== Data Filtering Summary ===")
    print(f"Total records matching criteria: {len(df)}")
    print("Harm Category Counts:")
    print(df['Harm_Category'].value_counts())
    
    print("\n=== Missing Human Control Distribution (Counts) ===")
    # Group by Harm Category and Missing Human Control
    group_counts = df.groupby(['Harm_Category', 'Missing_Human_Control']).size().unstack(fill_value=0)
    print(group_counts)

    # Create Contingency Table for Stats
    # Rows: Harm Category, Cols: Missing Human Control
    contingency_table = pd.crosstab(df['Harm_Category'], df['Missing_Human_Control'])
    print("\n=== Contingency Table ===")
    print(contingency_table)

    # Ensure 2x2 table for Fisher's Exact Test
    # Fisher's test requires a 2x2 matrix.
    # If one category has 0 missing controls, the shape might be wrong or the unstack might miss columns.
    # We force a 2x2 structure
    ct_2x2 = pd.DataFrame(
        index=['Safety_Reliability', 'Security'], 
        columns=[True, False]
    ).fillna(0)
    
    # Update with actual values
    for idx in ct_2x2.index:
        for col in ct_2x2.columns:
            try:
                ct_2x2.loc[idx, col] = contingency_table.loc[idx, col]
            except KeyError:
                ct_2x2.loc[idx, col] = 0
    
    print("\n=== 2x2 Matrix for Fisher's Test ===")
    print(ct_2x2)

    # Perform Statistical Test
    # Hypothesis: Safety/Reliability is associated with 'True' (Missing Human Control) more than Security is.
    # We use Fisher's Exact Test.
    odds_ratio, p_value = stats.fisher_exact(ct_2x2)

    print("\n=== Fisher's Exact Test Results ===")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(ct_2x2.astype(int), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Missing "Human-in-the-Loop" Controls by Harm Type')
    plt.ylabel('Harm Category')
    plt.xlabel('Missing Human Control Identified?')
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_enrichments.json
=== Data Filtering Summary ===
Total records matching criteria: 40
Harm Category Counts:
Harm_Category
Security              36
Safety_Reliability     4
Name: count, dtype: int64

=== Missing Human Control Distribution (Counts) ===
Missing_Human_Control  False  True 
Harm_Category                      
Safety_Reliability         4      0
Security                  33      3

=== Contingency Table ===
Missing_Human_Control  False  True 
Harm_Category                      
Safety_Reliability         4      0
Security                  33      3

=== 2x2 Matrix for Fisher's Test ===
                   True  False
Safety_Reliability     0     4
Security               3    33

=== Fisher's Exact Test Results ===
Odds Ratio: 0.0000
P-value: 1.0000


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap representing a **Contingency Table** (or Cross-tabulation).
*   **Purpose:** To visualize the frequency distribution of data across two categorical variables: "Harm Category" and whether "Missing Human Control" was identified. The color intensity corresponds to the count in each category intersection.

### 2. Axes
*   **Vertical (Y-axis):**
    *   **Title:** "Harm Category"
    *   **Labels:** The axis contains two categorical variables:
        1.  `Safety_Reliability`
        2.  `Security`
*   **Horizontal (X-axis):**
    *   **Title:** "Missing Human Control Identified?"
    *   **Labels:** The axis contains two boolean categorical variables:
        1.  `True`
        2.  `False`
*   **Value Ranges:** The axes represent categories rather than numerical ranges. The numerical values are contained within the matrix cells, ranging from a minimum of **0** to a maximum of **33**.

### 3. Data Trends
*   **High Values:** The area of highest density is the bottom-right quadrant, corresponding to the intersection of **Harm Category: Security** and **Missing Human Control Identified?: False**. This cell has a count of **33**, indicated by the dark blue color.
*   **Low Values:**
    *   The lowest value is found in the top-left quadrant (**Safety_Reliability** / **True**) with a count of **0**.
    *   Low values are also present in the **Safety_Reliability / False** cell (count: 4) and the **Security / True** cell (count: 3).
*   **Pattern:** There is a heavy skew toward the "Security" category where human control issues were *not* identified.

### 4. Annotations and Legends
*   **Title:** The chart is titled "Missing 'Human-in-the-Loop' Controls by Harm Type."
*   **Cell Annotations:** Each cell contains a number representing the exact count of observations for that specific intersection of categories.
*   **Color Mapping:** Although no separate legend key is provided, the plot uses a sequential color palette (shades of blue). Lighter shades (white/light blue) represent lower counts, while darker shades (deep blue) represent higher counts.

### 5. Statistical Insights
*   **Dominance of Security Issues:** The dataset is dominated by "Security" related harms. Out of the total observations shown (Sum = 40), **36** (90%) are related to Security, while only **4** (10%) are related to Safety_Reliability.
*   **Rarity of Identified Missing Controls:** It appears rare for "Missing Human Control" to be identified as a factor. Only **3** cases (7.5% of total) were flagged as `True`, and all of these were within the Security category.
*   **Safety_Reliability Outcomes:** There are zero instances where a "Safety_Reliability" harm was associated with a confirmed missing human control (`True`). All 4 cases in this category were marked `False`.
*   **Prevalence of False Negatives/Absence:** The majority of the data (37 out of 40 cases, or 92.5%) falls under the `False` column, suggesting that for most harms recorded here, missing human-in-the-loop controls were not identified as the issue.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
