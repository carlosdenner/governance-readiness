# Experiment 287: node_6_80

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_80` |
| **ID in Run** | 287 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:48:37.362899+00:00 |
| **Runtime** | 274.9s |
| **Parent** | `node_5_72` |
| **Children** | None |
| **Creation Index** | 288 |

---

## Hypothesis

> The Robustness Gap: Adversarial AI incidents exploit 'System Safety' competency
gaps significantly more frequently than 'Privacy' gaps.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8468 (Likely True) |
| **Posterior** | 0.7194 (Likely True) |
| **Surprise** | -0.1529 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 13.0 |
| Maybe True | 17.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 18.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Identify the dominant vulnerability class in adversarial attacks using ATLAS coding.

### Steps
- 1. Load `step3_incident_coding`.
- 2. Extract and explode the `competency_gaps` column (handle comma-separated strings).
- 3. Group gaps into categories based on keywords (e.g., 'Safety'/'Robustness' vs 'Privacy'/'Data').
- 4. Count frequencies.
- 5. Perform a Chi-Square Goodness of Fit test to see if Safety gaps are significantly more frequent than Privacy gaps.

### Deliverables
- Frequency bar chart of Competency Gaps; Chi-Square Goodness of Fit results.

---

## Analysis

The experiment was successfully executed but yielded a null result due to a
misalignment between the classification logic and the dataset's taxonomy. The
code analyzed 49 competency gaps from the `step3_incident_coding` table.
However, the keyword-based categorization strategy (scanning for terms like
'safety', 'robust', 'privacy') resulted in 0 matches for both the 'Safety' and
'Privacy' categories. Instead, 100% of the gaps (N=49) were classified as
'Other'. The debug output revealed that the actual data contains high-level
governance labels (e.g., 'Trust Readiness -- General Governance') rather than
the specific technical keywords anticipated by the programmer. Consequently, the
Chi-Square test could not be performed, and the hypothesis regarding the
prevalence of Safety vs. Privacy gaps could not be supported or refuted using
this specific keyword mapping. The result highlights that the ATLAS incidents in
this dataset are coded against governance bundles (Trust/Integration Readiness)
rather than direct technical failure modes.

---

## Review

The experiment was successfully executed, though the hypothesis could not be
statistically tested due to a taxonomy mismatch. The code correctly loaded the
`step3_incident_coding` dataset and parsed the `competency_gaps`. However, the
keyword-based classification logic (looking for terms like 'Safety',
'Robustness', 'Privacy') failed to match the actual data content, which consists
of high-level governance bundle names (e.g., 'Trust Readiness -- General
Governance') rather than specific technical failure modes. As a result, 100% of
the 49 analyzed gaps fell into the 'Other' category, rendering the Chi-Square
test impossible (0 vs 0). While the hypothesis was not supported, the experiment
successfully highlighted that the ATLAS incidents in this dataset are coded
against governance competencies rather than the technical vulnerability keywords
anticipated by the plan.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import os

def run_experiment():
    # Load Dataset
    filename = 'astalabs_discovery_all_data.csv'
    # Check parent dir first as per instructions
    path_parent = os.path.join('..', filename)
    path_current = filename
    
    fpath = None
    if os.path.exists(path_parent):
        fpath = path_parent
    elif os.path.exists(path_current):
        fpath = path_current
    else:
        print(f"Error: {filename} not found in current or parent directory.")
        return

    print(f"Loading data from: {fpath}")
    df = pd.read_csv(fpath, low_memory=False)

    # 1. Prepare ID Mapping from step1_sub_competencies
    # This helps if incidents use IDs (e.g., TR-1) instead of names
    sub_comp_df = df[df['source_table'] == 'step1_sub_competencies']
    id_map = {}
    if not sub_comp_df.empty:
        if 'id' in sub_comp_df.columns and 'name' in sub_comp_df.columns:
            for _, row in sub_comp_df.iterrows():
                if pd.notna(row['id']) and pd.notna(row['name']):
                    # map both "TR-1" and "tr-1" just in case
                    key = str(row['id']).strip()
                    val = str(row['name']).strip()
                    id_map[key] = val
                    id_map[key.lower()] = val

    # 2. Extract Incident Gaps
    # source_table: step3_incident_coding
    incidents = df[df['source_table'] == 'step3_incident_coding'].copy()
    
    if incidents.empty:
        print("No incident data found in step3_incident_coding.")
        return

    # Identify column
    gap_col = 'competency_gaps'
    if gap_col not in incidents.columns:
        if 'competency_gap' in incidents.columns:
            gap_col = 'competency_gap'
        else:
            print(f"Columns available: {incidents.columns.tolist()}")
            print("Could not find competency_gaps column.")
            return

    # 3. Process Gaps
    # Explode comma-separated strings
    raw_gaps = incidents[gap_col].dropna().astype(str)
    all_gaps = []
    
    for entry in raw_gaps:
        # Split
        tokens = [t.strip() for t in entry.split(',')]
        # Resolve IDs
        resolved_tokens = [id_map.get(t, id_map.get(t.lower(), t)) for t in tokens if t]
        all_gaps.extend(resolved_tokens)

    # 4. Categorize
    # Hypothesis: Safety/Robustness > Privacy
    safety_kw = ['safety', 'robust', 'reliab', 'secur', 'resilien', 'integr']
    privacy_kw = ['priva', 'confiden', 'data protect', 'anonym']
    
    counts = {'Safety': 0, 'Privacy': 0, 'Other': 0}
    
    # Debugging lists
    cat_debug = {'Safety': set(), 'Privacy': set(), 'Other': set()}

    for gap in all_gaps:
        txt = gap.lower()
        if any(k in txt for k in safety_kw):
            counts['Safety'] += 1
            cat_debug['Safety'].add(gap)
        elif any(k in txt for k in privacy_kw):
            counts['Privacy'] += 1
            cat_debug['Privacy'].add(gap)
        else:
            counts['Other'] += 1
            cat_debug['Other'].add(gap)

    # 5. Output Results
    print(f"Total Gaps Analyzed: {len(all_gaps)}")
    print(f"Safety/Robustness Count: {counts['Safety']}")
    print(f"Privacy Count: {counts['Privacy']}")
    print(f"Other Count: {counts['Other']}")
    
    print("\n--- Category Samples ---")
    print(f"Safety: {list(cat_debug['Safety'])[:5]}")
    print(f"Privacy: {list(cat_debug['Privacy'])[:5]}")
    print(f"Other: {list(cat_debug['Other'])[:5]}")

    # 6. Chi-Square Test
    # Compare Safety vs Privacy
    obs = [counts['Safety'], counts['Privacy']]
    if sum(obs) > 0:
        exp = [sum(obs)/2, sum(obs)/2]
        chi2, p = chisquare(obs, f_exp=exp)
        
        print(f"\nChi-Square Goodness of Fit (Safety vs Privacy):")
        print(f"Observed: {obs}")
        print(f"Expected: {exp}")
        print(f"Statistic: {chi2:.4f}, p-value: {p:.5f}")
        
        if p < 0.05:
            direction = "Safety > Privacy" if counts['Safety'] > counts['Privacy'] else "Privacy > Safety"
            print(f"Result: Significant difference ({direction}). Hypothesis supported.")
        else:
            print("Result: No significant difference. Hypothesis rejected.")
    else:
        print("No relevant gaps to test.")

    # 7. Plot
    labels = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['#d62728', '#1f77b4', '#7f7f7f'])
    plt.title('Competency Gaps in Adversarial Incidents (ATLAS)')
    plt.ylabel('Frequency')
    plt.bar_label(bars)
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading data from: astalabs_discovery_all_data.csv
Total Gaps Analyzed: 49
Safety/Robustness Count: 0
Privacy Count: 0
Other Count: 49

--- Category Samples ---
Safety: []
Privacy: []
Other: ['Trust Readiness -- General Governance']
No relevant gaps to test.


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided plot, here is the detailed breakdown:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot compares the frequency of different categories of "Competency Gaps" found in adversarial incidents, specifically within the context of the ATLAS framework (likely referring to MITRE ATLAS - Adversarial Threat Landscape for Artificial-Intelligence Systems).

### 2. Axes
*   **X-axis:**
    *   **Labels:** The axis represents categorical data with three distinct groups: "Safety", "Privacy", and "Other".
    *   **Title:** No explicit axis title is present, but the labels clearly denote incident categories.
*   **Y-axis:**
    *   **Title:** "Frequency".
    *   **Range:** The scale runs from 0 to 50, with major tick marks at intervals of 10 (0, 10, 20, 30, 40, 50).
    *   **Units:** Count (number of incidents).

### 3. Data Trends
*   **Tallest Bar:** The "Other" category is the dominant feature of the plot, reaching a frequency of 49.
*   **Shortest Bars:** Both the "Safety" and "Privacy" categories represent the lowest possible values, with no visible bars (frequency of 0).
*   **Pattern:** The data distribution is extremely skewed. 100% of the recorded competency gaps fall into the "Other" category, indicating a complete absence of gaps categorized specifically under Safety or Privacy in this specific dataset.

### 4. Annotations and Legends
*   **Data Labels:** There are numerical annotations placed directly above the bar locations to provide precise values:
    *   "0" above Safety.
    *   "0" above Privacy.
    *   "49" above Other.
*   **Title:** The chart is titled "Competency Gaps in Adversarial Incidents (ATLAS)".
*   **Legend:** There is no separate legend, as the x-axis labels sufficiently identify the data categories.

### 5. Statistical Insights
*   **Categorization Ambiguity:** The fact that all 49 incidents fall under "Other" suggests a potential lack of granularity in the classification system for this specific dataset. "Other" serves as a catch-all, masking the specific nature of the competency gaps.
*   **Exclusion of Traditional Concerns:** The data indicates that in the analyzed set of adversarial incidents, traditional "Safety" and "Privacy" failures were not the primary drivers or were not identified as the competency gap. This implies the incidents were likely related to other security pillars such as integrity, availability, or perhaps financial fraud, which are grouped into "Other".
*   **Concentration:** The dataset is monolithic; there is no variance across categories, as a single category accounts for the entire population size ($N=49$).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
