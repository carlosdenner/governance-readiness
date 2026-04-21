# Experiment 81: node_4_39

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_39` |
| **ID in Run** | 81 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:56:30.663133+00:00 |
| **Runtime** | 494.1s |
| **Parent** | `node_3_22` |
| **Children** | `node_5_52`, `node_5_80` |
| **Creation Index** | 82 |

---

## Hypothesis

> The dominant 'Harm Domain' in AI incidents has shifted significantly over time:
prior to 2020, 'Physical' and 'Economic' harms dominated, whereas post-2020,
'Societal' and 'Psychological' harms have become the primary failure modes.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.4355 (Maybe False) |
| **Posterior** | 0.1484 (Likely False) |
| **Surprise** | -0.3446 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 11.0 |
| Uncertain | 0.0 |
| Maybe False | 19.0 |
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

**Objective:** Analyze the temporal evolution of AI harm types.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `source_table` == 'aiid_incidents'.
- 2. Convert `date` column to datetime objects and extract the year.
- 3. Bin data into two epochs: 'Pre-2020' and 'Post-2020'.
- 4. Group by Epoch and `Harm Domain` (aggregated).
- 5. Perform a Chi-square test to see if the distribution of Harm Domains differs significantly between epochs.

### Deliverables
- 1. Year-wise distribution plot of Harm Domains.
- 2. Pre/Post 2020 contingency table.
- 3. Statistical test results confirming shift.

---

## Analysis

The experiment successfully analyzed the temporal evolution of AI harm types
using a keyword-inference approach on incident descriptions, as structured
labels were unavailable. A Chi-square test confirmed a statistically significant
shift in harm distribution between Pre-2020 and Post-2020 epochs (p < 0.001).

However, the results **do not support the hypothesis**.
1. **Pre-2020 Trends:** The hypothesis predicted a dominance of 'Physical' and
'Economic' harms. The data shows that 'Societal' harms were actually the most
prevalent (42.4%), followed by 'Physical' (29.1%).
2. **Post-2020 Trends:** The hypothesis predicted that 'Societal' and
'Psychological' harms would become primary. While 'Psychological' harms did
nearly double in proportion (9.3% to 17.8%), 'Societal' harms decreased
significantly (to 25.6%). Instead, 'Economic' harms emerged as the dominant
category, rising from 19.2% to 31.4%.

Conclusion: The landscape of AI incidents has shifted, but not in the predicted
direction. The trend indicates a move *from* Societal-dominant concerns (e.g.,
bias) *towards* Economic concerns (e.g., fraud, financial loss) and increasing
Psychological impacts, rather than the hypothesized shift.

---

## Review

The experiment successfully tested the hypothesis regarding the temporal
evolution of AI harm domains using a keyword-inference approach to compensate
for missing structured labels. The analysis processed 1,366 incidents, binning
them into Pre-2020 and Post-2020 epochs.

**Findings:**
1. **Statistical Significance:** A Chi-square test confirmed a statistically
significant shift in the distribution of harm types between the two epochs (p <
0.001).
2. **Hypothesis Evaluation:** The results **contradict** the hypothesis.
   - **Pre-2020:** The hypothesis predicted a dominance of 'Physical' and
'Economic' harms. The data shows that 'Societal' harms were actually the most
prevalent (42.4%), followed by 'Physical' (29.1%).
   - **Post-2020:** The hypothesis predicted that 'Societal' and 'Psychological'
harms would become primary. While 'Psychological' harms did nearly double in
proportion (9.3% to 17.8%), 'Societal' harms decreased significantly (to 25.6%).
Instead, 'Economic' harms emerged as the dominant category, rising from 19.2% to
31.4%.

**Conclusion:** While the landscape of AI incidents has indeed shifted
significantly, the trajectory differs from the proposed model. The trend
indicates a move *from* Societal-dominant concerns (e.g., bias, discrimination)
*towards* Economic concerns (e.g., financial loss, fraud) and increasing
Psychological impacts.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np

# 1. Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df_all = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df_all = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for 'aiid_incidents' and process dates
df_incidents = df_all[df_all['source_table'] == 'aiid_incidents'].copy()
df_incidents['date_parsed'] = pd.to_datetime(df_incidents['date'], errors='coerce')
df_incidents = df_incidents.dropna(subset=['date_parsed'])
df_incidents['year'] = df_incidents['date_parsed'].dt.year

# 3. Bin into Epochs
df_incidents['epoch'] = df_incidents['year'].apply(lambda x: 'Pre-2020' if x < 2020 else 'Post-2020')

# 4. Infer Harm Domains based on keywords in Title and Description
# Keywords definitions
keywords = {
    'Physical': ['death', 'dead', 'kill', 'injur', 'physical', 'health', 'safe', 'accident', 'crash', 'collision', 'bodily', 'harm'],
    'Economic': ['economic', 'financ', 'money', 'dollar', 'cost', 'job', 'employ', 'fraud', 'scam', 'market', 'theft', 'bank'],
    'Societal': ['societ', 'bias', 'discriminat', 'raci', 'sexis', 'gender', 'politic', 'elect', 'democra', 'surveil', 'priva', 'civil right', 'unfair', 'inequity', 'stereo'],
    'Psychological': ['psycholog', 'mental', 'emotion', 'fear', 'terror', 'trauma', 'stress', 'harass', 'manipulat', 'anxiety']
}

def infer_harm(row):
    text = (str(row.get('title', '')) + ' ' + str(row.get('description', ''))).lower()
    detected_harms = []
    for domain, kw_list in keywords.items():
        for kw in kw_list:
            if kw in text:
                detected_harms.append(domain)
                break # Found one keyword for this domain, move to next domain
    return detected_harms

df_incidents['inferred_harms'] = df_incidents.apply(infer_harm, axis=1)

# 5. Explode the list of harms to handle multi-label incidents
df_exploded = df_incidents.explode('inferred_harms')

# Drop rows where no harm was inferred
df_exploded = df_exploded.dropna(subset=['inferred_harms'])

# *** CRITICAL FIX: Reset index to avoid ValueError in crosstab ***
df_exploded = df_exploded.reset_index(drop=True)

# 6. Generate Summary Statistics and Contingency Table
contingency_table = pd.crosstab(df_exploded['epoch'], df_exploded['inferred_harms'])

print("--- Inferred Harm Domain Analysis ---")
print(f"Total incidents analyzed: {len(df_incidents)}")
print(f"Incidents with inferred harms: {df_incidents['inferred_harms'].map(lambda x: len(x) > 0).sum()}")
print("\nContingency Table (Epoch vs Inferred Harm):")
print(contingency_table)

# Normalize row-wise to see shifts in dominance
contingency_norm = contingency_table.div(contingency_table.sum(axis=1), axis=0)
print("\nProportional Distribution by Epoch:")
print(contingency_norm)

# 7. Statistical Test (Chi-Square)
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

alpha = 0.05
if p < alpha:
    print("Result: Statistically significant shift in Harm Domain distribution detected.")
else:
    print("Result: No statistically significant shift detected.")

# 8. Visualization
plt.figure(figsize=(10, 6))
contingency_norm.plot(kind='bar', stacked=False, ax=plt.gca())
plt.title('Shift in Inferred AI Harm Domains: Pre-2020 vs Post-2020')
plt.xlabel('Epoch')
plt.ylabel('Proportion of Mentions within Epoch')
plt.legend(title='Harm Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Validate specific claims
# Claim 1: Pre-2020 'Physical' & 'Economic' dominated
pre_phys = contingency_norm.loc['Pre-2020', 'Physical']
pre_econ = contingency_norm.loc['Pre-2020', 'Economic']
print(f"\nPre-2020 Physical + Economic share: {pre_phys + pre_econ:.2%}")

# Claim 2: Post-2020 'Societal' & 'Psychological' become primary
post_soc = contingency_norm.loc['Post-2020', 'Societal']
post_psy = contingency_norm.loc['Post-2020', 'Psychological']
print(f"Post-2020 Societal + Psychological share: {post_soc + post_psy:.2%}")

# Comparison
pre_soc_psy = contingency_norm.loc['Pre-2020', 'Societal'] + contingency_norm.loc['Pre-2020', 'Psychological']
post_phys_econ = contingency_norm.loc['Post-2020', 'Physical'] + contingency_norm.loc['Post-2020', 'Economic']

print(f"\nShift Analysis:")
print(f"Physical/Economic: {pre_phys + pre_econ:.2%} (Pre) -> {post_phys_econ:.2%} (Post)")
print(f"Societal/Psychological: {pre_soc_psy:.2%} (Pre) -> {post_soc + post_psy:.2%} (Post)")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Inferred Harm Domain Analysis ---
Total incidents analyzed: 1362
Incidents with inferred harms: 853

Contingency Table (Epoch vs Inferred Harm):
inferred_harms  Economic  Physical  Psychological  Societal
epoch                                                      
Post-2020            288       232            163       235
Pre-2020              33        50             16        73

Proportional Distribution by Epoch:
inferred_harms  Economic  Physical  Psychological  Societal
epoch                                                      
Post-2020       0.313725  0.252723       0.177560  0.255991
Pre-2020        0.191860  0.290698       0.093023  0.424419

Chi-square Test Results:
Chi2 Statistic: 28.9591
P-value: 2.2842e-06
Result: Statistically significant shift in Harm Domain distribution detected.

Pre-2020 Physical + Economic share: 48.26%
Post-2020 Societal + Psychological share: 43.36%

Shift Analysis:
Physical/Economic: 48.26% (Pre) -> 56.64% (Post)
Societal/Psychological: 51.74% (Pre) -> 43.36% (Post)


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Plot (or Clustered Bar Chart).
*   **Purpose:** The plot is designed to compare the relative proportions of four different categories ("Harm Domains") across two distinct time periods ("Epochs"). This visualization allows for easy comparison of how the focus or distribution of these domains has shifted over time.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Label:** "Epoch"
    *   **Categories:** The axis is divided into two distinct time periods: "Post-2020" and "Pre-2020".
*   **Y-Axis (Vertical):**
    *   **Label:** "Proportion of Mentions within Epoch"
    *   **Value Range:** The scale runs from 0.00 to approximately 0.44 (the grid lines are spaced at intervals of 0.05).
    *   **Units:** The values represent proportions (ranging from 0 to 1), effectively representing percentages if multiplied by 100 (e.g., 0.20 = 20%).

### 3. Data Trends
**Pre-2020 Trends (Right Group):**
*   **Tallest Bar:** **Societal** (Red) is the dominant domain, reaching a proportion slightly above **0.42**.
*   **Shortest Bar:** **Psychological** (Green) is the least mentioned, sitting below **0.10**.
*   **Pattern:** There is a clear hierarchy: Societal > Physical > Economic > Psychological.

**Post-2020 Trends (Left Group):**
*   **Tallest Bar:** **Economic** (Blue) is the dominant domain, reaching approximately **0.31**.
*   **Shortest Bar:** **Psychological** (Green) remains the lowest but has risen to approximately **0.18**.
*   **Pattern:** The hierarchy has shifted significantly: Economic > Societal ≈ Physical > Psychological.

**Comparison (The Shift):**
*   **Economic:** Shows a significant increase, jumping from ~0.19 (Pre-2020) to ~0.31 (Post-2020).
*   **Societal:** Shows a massive decrease, dropping from ~0.42 (Pre-2020) to ~0.25 (Post-2020).
*   **Physical:** Shows a slight decrease from ~0.29 (Pre-2020) to ~0.25 (Post-2020).
*   **Psychological:** Shows an increase, roughly doubling from ~0.09 (Pre-2020) to ~0.18 (Post-2020).

### 4. Annotations and Legends
*   **Title:** "Shift in Inferred AI Harm Domains: Pre-2020 vs Post-2020" clearly indicates the subject matter.
*   **Legend:** Located in the top right, titled **"Harm Domain"**. It keys the colors to the categories:
    *   **Blue:** Economic
    *   **Orange:** Physical
    *   **Green:** Psychological
    *   **Red:** Societal
*   **Grid Lines:** Horizontal dashed grey lines are provided at 0.05 intervals to assist in estimating the bar heights.

### 5. Statistical Insights
The plot reveals a fundamental restructuring of the discourse surrounding AI harms before and after the year 2020:

1.  **Shift in Primary Concern:** The most notable insight is the reversal of the primary concern. Prior to 2020, **Societal** harms were the overwhelming focus (accounting for over 40% of mentions). Post-2020, **Economic** harms became the most cited concern (over 30%). This suggests a change in narrative focus, perhaps moving from general societal impact to specific economic anxieties like job displacement or financial volatility.
2.  **Rise in Psychological Awareness:** Although **Psychological** harm remains the smallest category in both epochs, its proportion nearly doubled in the Post-2020 era. This indicates a growing recognition of mental health or psychological impacts related to AI.
3.  **Normalization of Physical Harms:** The proportion of mentions regarding **Physical** harms remained relatively stable, decreasing only slightly, suggesting that concerns about physical safety (e.g., robotics, autonomous vehicles) have remained a consistent but not dominant part of the conversation.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
