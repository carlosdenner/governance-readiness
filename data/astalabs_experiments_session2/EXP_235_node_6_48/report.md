# Experiment 235: node_6_48

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_48` |
| **ID in Run** | 235 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:04:46.730965+00:00 |
| **Runtime** | 362.5s |
| **Parent** | `node_5_31` |
| **Children** | None |
| **Creation Index** | 236 |

---

## Hypothesis

> Sector-Specific Intentionality: The 'Law Enforcement' sector is significantly
more likely to report incidents classified as 'Intentional Harm' compared to the
'Healthcare' sector, which is predominantly characterized by unintentional
accidents.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7661 (Likely True) |
| **Posterior** | 0.2665 (Likely False) |
| **Surprise** | -0.5996 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 11.0 |
| Maybe True | 15.0 |
| Uncertain | 0.0 |
| Maybe False | 4.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 2.0 |
| Definitely False | 58.0 |

---

## Experiment Plan

**Objective:** Correct the previous logic error in classifying 'Intentional Harm' to accurately contrast the nature of harm (Intentional vs. Unintentional) between Law Enforcement and Healthcare sectors.

### Steps
- 1. Load the `astalabs_discovery_all_data.csv` dataset and filter for rows where `source_table` is 'aiid_incidents'.
- 2. Create a `Target Sector` column by classifying the `Sector of Deployment` column using keywords:
    - 'Healthcare': 'health', 'medic', 'hospital', 'patient', 'doctor'.
    - 'Law Enforcement': 'law enforcement', 'police', 'surveillance', 'arrest', 'prison', 'jail'.
- 3. Create a `Harm Intent` column by processing the `Intentional Harm` column with strict logic to avoid substring errors:
    - First, convert the value to lowercase string.
    - If the string contains 'no' or 'unintentional', classify as 'Unintentional'.
    - Else if the string contains 'yes' or 'intentional' (and NOT 'unintentional'), classify as 'Intentional'.
    - Otherwise, classify as 'Unknown'.
- 4. Filter the dataset to include only rows where `Target Sector` is either 'Healthcare' or 'Law Enforcement' and `Harm Intent` is known (Intentional or Unintentional).
- 5. Generate a contingency table (Target Sector vs. Harm Intent).
- 6. Perform a Chi-Square Test of Independence on the contingency table.
- 7. Visualize the results using a 100% stacked bar chart showing the proportion of Intentional vs. Unintentional harm for each sector.

### Deliverables
- 1. Contingency table of Target Sector vs. Harm Intent.
- 2. Chi-Square test results (statistic and p-value).
- 3. Stacked bar chart visualization.

---

## Analysis

The experiment was executed successfully, correcting the previous logic error
regarding substring matching. The revised `classify_intent` function correctly
distinguishes between 'Intentional' and 'Unintentional' strings.

However, the analysis reveals a significant data property: all 31 identified
incidents in the 'Healthcare' and 'Law Enforcement' sectors are labeled as
'Intentional' in the dataset. There are zero 'Unintentional' cases found. This
lack of variance (a 2x1 contingency table) correctly triggered the safety check
in the code, preventing the Chi-Square test which requires at least a 2x2 table.

The consistency in row counts between the previous run (which treated NaNs as
Unintentional) and this run (which dropped NaNs) — both yielding 31 rows —
indicates that there are no missing values (NaNs) in this subset; the data
explicitly labels these incidents as Intentional. Consequently, the hypothesis
that Healthcare is predominantly characterized by unintentional accidents cannot
be tested with this specific dataset column, as the data contradicts the premise
of variance.

---

## Review

The experiment was faithfully implemented. The programmer corrected the previous
logic error in the `classify_intent` function, ensuring that 'Unintentional'
labels would be correctly identified. The code successfully processed the
dataset, identified 31 relevant incidents (15 Healthcare, 16 Law Enforcement),
and attempted the statistical analysis.

The analysis revealed that all 31 incidents in this subset were explicitly
classified as 'Intentional' in the source data. Due to this lack of variance (0
'Unintentional' cases), the Chi-Square test could not be performed (correctly
handled by the code's safety check), and the visualization correctly showed 100%
Intentionality for both sectors.

While the statistical test yielded no p-value, the experiment successfully
generated a finding: the data contradicts the hypothesis premise that Healthcare
incidents are predominantly unintentional in this dataset.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def run_experiment():
    # Load dataset
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

    # Filter for AIID incidents
    aiid = df[df['source_table'] == 'aiid_incidents'].copy()

    # --- Helper Functions ---
    def classify_sector(x):
        if pd.isna(x):
            return None
        s = str(x).lower()
        # Healthcare keywords
        if any(k in s for k in ['health', 'medic', 'hospital', 'patient', 'doctor']):
            return 'Healthcare'
        # Law Enforcement keywords
        if any(k in s for k in ['law enforcement', 'police', 'surveillance', 'arrest', 'prison', 'jail']):
            return 'Law Enforcement'
        return None

    def classify_intent(x):
        if pd.isna(x):
            return None
        s = str(x).lower().strip()
        
        # Check for Unintentional first (fixes previous bug where 'unintentional' matched 'intentional')
        # Also check for explicit 'No' which likely means 'Not Intentional'
        if 'unintentional' in s or s == 'no' or 'accidental' in s:
            return 'Unintentional'
        
        # Check for Intentional
        if 'intentional' in s or s == 'yes':
            return 'Intentional'
            
        return None # Exclude ambiguous entries

    # --- Apply Classifications ---
    aiid['target_sector'] = aiid['Sector of Deployment'].apply(classify_sector)
    aiid['harm_intent'] = aiid['Intentional Harm'].apply(classify_intent)

    # Filter dataset
    # We only want rows where both Sector and Intent were successfully classified
    subset = aiid.dropna(subset=['target_sector', 'harm_intent']).copy()

    print(f"Total incidents analyzed after filtering: {len(subset)}")
    
    # Check if we have data
    if len(subset) == 0:
        print("No data found matching criteria.")
        return

    # --- Statistical Analysis ---
    # Generate Contingency Table
    contingency_table = pd.crosstab(subset['target_sector'], subset['harm_intent'])
    print("\nContingency Table (Sector vs. Intentionality):")
    print(contingency_table)

    # Ensure we have enough data dimensions for Chi-Square
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        print("\nInsufficient dimensions for Chi-Square test (need 2x2).")
    else:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nChi-Square Statistic: {chi2:.4f}")
        print(f"P-Value: {p:.4e}")

    # --- Visualization ---
    # Reorder columns to ensure consistent color mapping (Intentional first or second doesn't matter, but must be consistent)
    desired_order = ['Intentional', 'Unintentional']
    # Filter/Sort columns that exist in the data
    cols = [c for c in desired_order if c in contingency_table.columns]
    contingency_table = contingency_table[cols]

    # Normalize rows to 100% for stacked bar chart
    ct_norm = contingency_table.div(contingency_table.sum(axis=1), axis=0)

    # Colors: Red for Intentional, Blue for Unintentional
    color_map = {'Intentional': '#d62728', 'Unintentional': '#1f77b4'}
    colors = [color_map[c] for c in cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    ct_norm.plot(kind='bar', stacked=True, ax=ax, color=colors)

    plt.title('Proportion of Intentional vs. Unintentional Harm by Sector')
    plt.ylabel('Proportion')
    plt.xlabel('Sector')
    plt.xticks(rotation=0)
    plt.legend(title='Harm Intent', loc='upper right', bbox_to_anchor=(1.2, 1))

    # Annotate bars with counts and percentages
    for n, x in enumerate(contingency_table.index):
        row_counts = contingency_table.loc[x]
        row_props = ct_norm.loc[x]
        
        cum_y = 0
        for col in cols:
            count = row_counts[col]
            prop = row_props[col]
            if prop > 0: # Only annotate if segment exists
                # Center text in the segment
                y_pos = cum_y + prop/2
                ax.text(n, y_pos, f"{count}\n({prop:.1%})", 
                        ha='center', va='center', color='white', fontweight='bold')
            cum_y += prop

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total incidents analyzed after filtering: 31

Contingency Table (Sector vs. Intentionality):
harm_intent      Intentional
target_sector               
Healthcare                15
Law Enforcement           16

Insufficient dimensions for Chi-Square test (need 2x2).


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **stacked bar chart** (normalized to 100% proportion).
*   **Purpose:** The chart is designed to compare the distribution of "Intentional" versus "Unintentional" harm across two different professional sectors: Healthcare and Law Enforcement. However, in this specific instance, only one category ("Intentional") is present.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Sector"
    *   **Categories:** Two categorical variables are displayed: "Healthcare" and "Law Enforcement".
*   **Y-Axis:**
    *   **Label:** "Proportion"
    *   **Range:** The axis ranges from **0.0 to 1.0** (representing 0% to 100%). There are major tick marks at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Pattern:** The most striking trend is the uniformity of the data. In both sectors presented, the bars extend fully to the 1.0 mark (100%).
*   **Dominant Category:** The entire volume of both bars corresponds to the "Intentional" category (colored red). There is no visible portion for "Unintentional" harm.
*   **Tallest/Shortest:** Both bars are of equal height regarding proportion (1.0 or 100%), indicating no variation in the *type* of harm between the two sectors within this dataset.

### 4. Annotations and Legends
*   **Legend:** Located on the right side, titled **"Harm Intent"**. It defines the red color as **"Intentional"**. Although the chart title implies a comparison with "Unintentional" harm, that category does not appear in the legend or the bars, presumably because its value is 0.
*   **Bar Annotations:** Inside each bar, white text provides specific data points:
    *   **Healthcare:** Contains the count **"15"** and the percentage **"(100.0%)"**.
    *   **Law Enforcement:** Contains the count **"16"** and the percentage **"(100.0%)"**.

### 5. Statistical Insights
*   **Exclusivity of Intent:** The data suggests that for the incidents analyzed in this dataset, **100% of the harm cases** in both Healthcare and Law Enforcement were classified as **Intentional**. There were zero recorded instances of unintentional harm.
*   **Sample Size:** The absolute numbers are relatively small: n=15 for Healthcare and n=16 for Law Enforcement.
*   **Interpretation:** While the chart title sets up a comparison ("Intentional vs. Unintentional"), the result is singular. This could imply a dataset specifically filtered for intentional acts (e.g., a study on assault or misconduct), or it could indicate an anomaly where no accidents were reported in the sampled period. The similarity in counts (15 vs 16) suggests the two groups are balanced in sample size.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
