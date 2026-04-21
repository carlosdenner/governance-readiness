# Experiment 118: node_4_47

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_47` |
| **ID in Run** | 118 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:37:19.438555+00:00 |
| **Runtime** | 321.3s |
| **Parent** | `node_3_14` |
| **Children** | `node_5_57`, `node_5_81` |
| **Creation Index** | 119 |

---

## Hypothesis

> Adversarial Tech Sophistication: In ATLAS case studies, the 'Exfiltration'
tactic is significantly more likely to co-occur with 'Evasion' tactics than
'Impact' (Destruction) tactics are, suggesting that data theft attacks rely on
stealth/bypass mechanisms more than disruption attacks do.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4066 (Maybe False) |
| **Surprise** | -0.4024 |
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
| Maybe False | 56.0 |
| Definitely False | 4.0 |

---

## Experiment Plan

**Objective:** Analyze tactic co-occurrence in adversarial AI attacks to understand attack chains.

### Steps
- 1. Load 'atlas_cases' subset.
- 2. Parse the '92_tactics' column (likely semicolon- or comma-separated strings).
- 3. One-hot encode the presence of 'Exfiltration', 'Evasion', and 'Impact' tactics for each case.
- 4. Create two groups: Cases with Exfiltration vs. Cases with Impact.
- 5. Compare the rate of 'Evasion' presence between these two groups using a Fisher's Exact Test.
- 6. Calculate the Jaccard similarity index between Exfiltration and Evasion.

### Deliverables
- Co-occurrence matrix; Fisher's Exact Test p-value; Jaccard similarity score.

---

## Analysis

The experiment successfully analyzed the relationship between attack objectives
('Exfiltration' vs. 'Impact') and the use of 'Evasion' tactics in adversarial AI
incidents. The dataset was successfully filtered, identifying 52 relevant cases
with populated tactic data (though 104 rows were processed, only those with
valid tactic strings contributed to the findings).

The results **contradict the hypothesis** that Exfiltration attacks rely more on
Evasion than Impact attacks do.

1.  **Observed Rates**: Attacks involving 'Impact' (destruction/disruption) had
a higher co-occurrence rate with 'Evasion' tactics (38.89%, 14/36 cases)
compared to 'Exfiltration' attacks (33.33%, 5/15 cases).
2.  **Statistical Significance**: The Fisher's Exact Test (testing for
Exfiltration > Impact) yielded a p-value of 0.7529, failing to reject the null
hypothesis. The trend actually leans in the opposite direction.
3.  **Jaccard Similarity**: The similarity between Exfiltration and Evasion was
low (0.1515), indicating they are not strongly coupled in this dataset.

These findings suggest that in the MITRE ATLAS dataset, adversarial attacks
aiming for destruction are just as likely (if not slightly more so) to employ
stealth and evasion techniques as those aiming for data theft.

---

## Review

The experiment was successfully executed and the code produced the necessary
metrics to evaluate the hypothesis. Note: The script processed 104 rows by
combining 'atlas_cases' and 'step3_incident_coding', likely resulting in
duplicate entries for the 52 underlying cases. However, since the observed trend
(Impact > Exfiltration) contradicts the hypothesis (Exfiltration > Impact), and
the p-value (0.75) is far from significant even with the potentially inflated
sample size, the conclusion to reject the hypothesis remains statistically
robust. Correcting for duplicates would only further weaken the statistical
significance, confirming the null result.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_experiment():
    # 1. Load Dataset
    filename = 'astalabs_discovery_all_data.csv'
    if os.path.exists(filename):
        filepath = filename
    elif os.path.exists(os.path.join('..', filename)):
        filepath = os.path.join('..', filename)
    else:
        print(f"Dataset {filename} not found.")
        return

    print(f"Loading {filepath}...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"Failed to load csv: {e}")
        return

    # 2. Filter for ATLAS cases
    # The metadata indicates 'atlas_cases' or 'step3_incident_coding'.
    # We prefer 'step3_incident_coding' as it likely contains the coded analysis, 
    # but 'atlas_cases' is the raw source. We'll check both.
    
    # Let's try to get a consolidated view or pick the best source.
    # Looking at previous exploration, 'step3_incident_coding' has 52 rows and 'tactics' column.
    
    atlas_df = df[df['source_table'].isin(['atlas_cases', 'step3_incident_coding'])].copy()
    
    if atlas_df.empty:
        print("No ATLAS case rows found.")
        return

    # 3. Identify the correct tactics column
    # We explicitly exclude 'n_tactics' (count) and look for string columns.
    potential_cols = ['tactics', 'tactics_used', '92_tactics'] # 92_tactics from metadata description
    
    target_col = None
    for col in potential_cols:
        if col in atlas_df.columns:
            # Check if it has non-null values and is object/string type
            valid_rows = atlas_df[col].dropna()
            if not valid_rows.empty:
                # Check if values look like strings (not numbers)
                sample = str(valid_rows.iloc[0])
                if not sample.isdigit():
                    target_col = col
                    break
    
    # Fallback: search all columns with 'tactic' in name if specific ones fail
    if not target_col:
        cols = [c for c in atlas_df.columns if 'tactic' in c.lower() and 'n_' not in c.lower()]
        for col in cols:
             valid_rows = atlas_df[col].dropna()
             if not valid_rows.empty and not str(valid_rows.iloc[0]).isdigit():
                 target_col = col
                 break

    if not target_col:
        print("Could not find a valid string column for tactics.")
        print("Available columns:", atlas_df.columns.tolist())
        return

    print(f"Using column '{target_col}' for analysis (n={atlas_df[target_col].notna().sum()}).")
    
    # 4. Parse Tactics
    # Normalize to lowercase and split
    def parse_tactics(val):
        if pd.isna(val):
            return []
        val = str(val).lower()
        # Replace common delimiters
        val = val.replace(';', ',').replace('/', ',')
        tokens = [t.strip() for t in val.split(',')]
        return tokens

    atlas_df['parsed_tactics'] = atlas_df[target_col].apply(parse_tactics)

    # 5. Create Indicators
    # We look for keywords: 'exfiltration', 'evasion' (often 'defense evasion'), 'impact'
    # Note: MITRE ATLAS uses 'Defense Evasion', so we search for 'evasion'.
    # 'Impact' is a top-level tactic.
    # 'Exfiltration' is a top-level tactic.
    
    atlas_df['has_exfil'] = atlas_df['parsed_tactics'].apply(lambda x: any('exfiltration' in t for t in x))
    atlas_df['has_evasion'] = atlas_df['parsed_tactics'].apply(lambda x: any('evasion' in t for t in x))
    atlas_df['has_impact'] = atlas_df['parsed_tactics'].apply(lambda x: any('impact' in t for t in x))

    # 6. Analysis Groups
    group_exfil = atlas_df[atlas_df['has_exfil']]
    group_impact = atlas_df[atlas_df['has_impact']]
    
    n_exfil = len(group_exfil)
    n_impact = len(group_impact)
    
    print(f"Total analyzed cases: {len(atlas_df)}")
    print(f"Cases with Exfiltration: {n_exfil}")
    print(f"Cases with Impact: {n_impact}")

    if n_exfil == 0 or n_impact == 0:
        print("Insufficient data in one or both groups to compare.")
        return

    # Count Evasion in each group
    k_exfil_evasion = group_exfil['has_evasion'].sum()
    k_impact_evasion = group_impact['has_evasion'].sum()
    
    rate_exfil = k_exfil_evasion / n_exfil
    rate_impact = k_impact_evasion / n_impact
    
    print(f"\n--- Co-occurrence Results ---")
    print(f"Exfiltration Cases: {k_exfil_evasion}/{n_exfil} ({rate_exfil:.2%}) also use Evasion")
    print(f"Impact Cases:       {k_impact_evasion}/{n_impact} ({rate_impact:.2%}) also use Evasion")

    # Fisher's Exact Test
    # Table:
    #             | Evasion+ | Evasion-
    # Exfil Group |    a     |    b
    # Impact Group|    c     |    d
    
    a = k_exfil_evasion
    b = n_exfil - k_exfil_evasion
    c = k_impact_evasion
    d = n_impact - k_impact_evasion
    
    odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
    
    print(f"\nFisher's Exact Test (H1: Exfil > Impact for Evasion co-occurrence)")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Jaccard Similarity (Exfiltration, Evasion)
    # Intersection = Cases with BOTH Exfil AND Evasion
    # Union = Cases with EITHER Exfil OR Evasion
    n_inter = atlas_df[atlas_df['has_exfil'] & atlas_df['has_evasion']].shape[0]
    n_union = atlas_df[atlas_df['has_exfil'] | atlas_df['has_evasion']].shape[0]
    jaccard = n_inter / n_union if n_union > 0 else 0
    
    print(f"\nJaccard Similarity (Exfiltration <-> Evasion): {jaccard:.4f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Exfiltration Cases', 'Impact Cases'], [rate_exfil, rate_impact], 
                   color=['#4c72b0', '#c44e52'])
    plt.ylabel('Proportion Involving Evasion Tactics')
    plt.title('Adversarial Tech Sophistication: Stealth vs. Destruction')
    plt.ylim(0, 1.1)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.1%}', ha='center', va='bottom')
    
    plt.show()

if __name__ == '__main__':
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading astalabs_discovery_all_data.csv...
Using column 'tactics' for analysis (n=52).
Total analyzed cases: 104
Cases with Exfiltration: 15
Cases with Impact: 36

--- Co-occurrence Results ---
Exfiltration Cases: 5/15 (33.33%) also use Evasion
Impact Cases:       14/36 (38.89%) also use Evasion

Fisher's Exact Test (H1: Exfil > Impact for Evasion co-occurrence)
Odds Ratio: 0.7857
P-value: 0.7529

Jaccard Similarity (Exfiltration <-> Evasion): 0.1515


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the analysis:

**1. Plot Type**
*   **Type:** Bar Chart.
*   **Purpose:** The plot compares a numerical metric ("Proportion Involving Evasion Tactics") across two distinct categories of adversarial cases ("Exfiltration Cases" and "Impact Cases"). It is designed to contrast the sophistication (specifically the use of stealth/evasion) between attacks focused on data theft versus those focused on destruction.

**2. Axes**
*   **X-axis:**
    *   **Labels:** The axis features two categorical labels: "Exfiltration Cases" and "Impact Cases".
    *   **Range:** Two discrete categories.
*   **Y-axis:**
    *   **Title:** "Proportion Involving Evasion Tactics".
    *   **Range:** The numerical scale runs from **0.0 to 1.0** (representing 0% to 100%), with tick marks every 0.2 units. The visible plot area extends slightly beyond 1.0 to approximately 1.1.
    *   **Units:** The axis uses decimal proportions, while the data labels convert these to percentages.

**3. Data Trends**
*   **Tallest Bar:** The **"Impact Cases"** (red bar) is the tallest, indicating a higher prevalence of evasion tactics in this category.
*   **Shortest Bar:** The **"Exfiltration Cases"** (blue bar) is the shortest.
*   **Pattern:** There is a visible, though not massive, difference between the two categories. Counter to what might be assumed (that data theft requires more stealth), the data shows that destructive/impact-oriented cases actually utilize evasion tactics more frequently in this dataset.

**4. Annotations and Legends**
*   **Title:** "Adversarial Tech Sophistication: Stealth vs. Destruction" sets the context of the comparison.
*   **Data Labels:**
    *   Above the "Exfiltration Cases" bar: **33.3%**
    *   Above the "Impact Cases" bar: **38.9%**
    *   These annotations provide the exact values, making it easier to read than estimating from the Y-axis.
*   **Color Coding:** The bars are color-coded (Blue for Exfiltration, Red for Impact) to visually distinguish the two categories, though there is no separate legend box.

**5. Statistical Insights**
*   **Prevalence of Evasion:** Evasion tactics are a significant component of both attack types, appearing in over one-third of all cases for both categories.
*   **Comparative Insight:** Impact cases involve evasion tactics at a higher rate (**38.9%**) compared to Exfiltration cases (**33.3%**). This suggests that adversaries aiming for destruction or disruption (Impact) are slightly more likely to employ sophisticated evasion techniques than those aiming solely to steal data (Exfiltration).
*   **Sophistication Gap:** The gap between the two is **5.6 percentage points**. While Exfiltration implies a need for "Stealth," the data suggests that "Destruction" campaigns are actually more sophisticated regarding evasion, perhaps to ensure the payload is delivered and executed before detection.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
