# Experiment 69: node_4_31

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_31` |
| **ID in Run** | 69 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:19:46.468408+00:00 |
| **Runtime** | 255.1s |
| **Parent** | `node_3_19` |
| **Children** | `node_5_54`, `node_5_94` |
| **Creation Index** | 70 |

---

## Hypothesis

> Autonomy-Redress Compensatory Mechanism: High-autonomy systems are significantly
more likely to have formal 'Appeal Processes' established than low-autonomy
systems, suggesting agencies use human-in-the-loop as a substitute for formal
redress.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.2581 (Likely False) |
| **Posterior** | 0.2418 (Likely False) |
| **Surprise** | -0.0196 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 30.0 |
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

**Objective:** Test if higher autonomy correlates with the presence of formal appeal mechanisms using Fisher's Exact Test due to small sample sizes.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `source_table == 'eo13960_scored'`.
- 2. Create `autonomy_bin`: Map 'Yes - All individual...' and 'Other – Immediate...' to 'High'; Map 'No - Some individual...' to 'Low'. Drop others.
- 3. Create `appeal_bin`: Map 'Yes' to 1; Map 'No – it is not operationally...' and 'No – Law...' to 0. Drop others.
- 4. Drop rows with NaN in either new column.
- 5. Generate a contingency table (Autonomy x Appeal).
- 6. Perform Fisher's Exact Test to determine statistical significance.
- 7. Calculate proportions of Appeal=1 for High vs Low autonomy groups.

### Deliverables
- Contingency table; Fisher's Exact Test p-value; Bar plot comparing appeal process rates by autonomy level.

---

## Analysis

The experiment successfully tested the 'Autonomy-Redress Compensatory Mechanism'
hypothesis using Fisher's Exact Test due to the small sample size of high-
autonomy systems.

The results **do not support** the hypothesis that high-autonomy systems are
more likely to have formal appeal processes.

1. **Directionality**: Contrary to the hypothesis, 'High' autonomy systems
showed a *lower* rate of established appeal processes (40.0%, n=2/5) compared to
'Low' autonomy systems (56.1%, n=74/132).

2. **Statistical Significance**: The difference was not statistically
significant (Fisher's Exact Test p=0.656), indicating that autonomy level is not
a strong predictor of redress availability in this dataset.

3. **limitations**: The analysis is heavily constrained by the extreme scarcity
of high-autonomy systems in the federal inventory (only 5 valid cases vs 132
low-autonomy cases), making the 40% figure statistically fragile.

---

## Review

The experiment was executed faithfully according to the plan. The programmer
correctly handled the sparse data issues identified in the debug phase by
implementing robust text mapping for 'Autonomy Level' and 'Appeal Process' and
utilizing Fisher's Exact Test to accommodate the small sample size of high-
autonomy systems (n=5).

**Findings:**
The results **reject** the hypothesis that high-autonomy systems are more likely
to have formal appeal processes.
1.  **Inverse Trend:** High-autonomy systems actually demonstrated a *lower*
rate of appeal process availability (40.0%, n=2/5) compared to low-autonomy
systems (56.1%, n=74/132).
2.  **No Statistical Significance:** The difference was not statistically
significant (Fisher's Exact Test p=0.66), indicating that autonomy level is not
currently a predictor of redress governance in the federal inventory.
3.  **Conclusion:** The data does not support the existence of a 'compensatory
mechanism' where agencies bolster high-automation systems with stronger appeal
rights. Instead, the higher prevalence of appeals in low-autonomy systems
suggests that standard, human-supervised workflows are currently more mature in
their procedural rights infrastructure.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Load data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
    
    # 2. Map Autonomy Level
    # High: Fully automated or immediate intervention not practicable
    high_autonomy_markers = [
        'Yes - All individual decisions or actions are automated',
        'Other – Immediate human intervention is not practicable'
    ]
    low_autonomy_markers = [
        'No - Some individual decisions or actions require direct human oversight'
    ]
    
    def map_autonomy(val):
        if pd.isna(val):
            return None
        val_str = str(val)
        if any(marker in val_str for marker in high_autonomy_markers):
            return 'High'
        if any(marker in val_str for marker in low_autonomy_markers):
            return 'Low'
        return None

    eo_df['autonomy_level'] = eo_df['57_autonomous_impact'].apply(map_autonomy)

    # 3. Map Appeal Process
    # Yes: Explicitly 'Yes'
    # No: Explicit denials or waivers
    def map_appeal(val):
        if pd.isna(val):
            return None
        val_str = str(val)
        if val_str.strip() == 'Yes':
            return 'Yes'
        if 'No –' in val_str or 'waived' in val_str:
            return 'No'
        return None

    eo_df['has_appeal'] = eo_df['65_appeal_process'].apply(map_appeal)

    # 4. Filter for valid data
    valid_df = eo_df.dropna(subset=['autonomy_level', 'has_appeal'])
    
    print(f"Data points after filtering: {len(valid_df)}")
    
    # 5. Create Contingency Table
    # We want rows to be Autonomy (High/Low) and columns to be Appeal (Yes/No)
    contingency = pd.crosstab(valid_df['autonomy_level'], valid_df['has_appeal'])
    
    # Ensure both rows/cols exist even if counts are 0
    for level in ['High', 'Low']:
        if level not in contingency.index:
            contingency.loc[level] = [0, 0]
    for response in ['Yes', 'No']:
        if response not in contingency.columns:
            contingency[response] = 0
            
    # Reorder for consistency: Rows=[High, Low], Cols=[Yes, No]
    contingency = contingency.loc[['High', 'Low'], ['Yes', 'No']]
    
    print("\n--- Contingency Table (Autonomy vs Appeal) ---")
    print(contingency)
    
    # 6. Statistical Test (Fisher's Exact Test due to small sample sizes)
    odds_ratio, p_value = stats.fisher_exact(contingency)
    
    print("\n--- Statistical Results ---")
    print(f"Fisher's Exact Test p-value: {p_value:.4f}")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    
    # Calculate percentages for plotting
    rates = contingency.div(contingency.sum(axis=1), axis=0)['Yes'] * 100
    
    print("\n--- Appeal Process Rates ---")
    print(rates)

    # 7. Visualization
    plt.figure(figsize=(8, 6))
    bars = plt.bar(rates.index, rates.values, color=['#d62728', '#1f77b4'], alpha=0.7)
    plt.title('Availability of Appeal Process by System Autonomy Level')
    plt.ylabel('Percentage with Formal Appeal Process (%)')
    plt.xlabel('Autonomy Level')
    plt.ylim(0, 100)
    
    # Add counts to bars
    for bar, label in zip(bars, rates.index):
        height = bar.get_height()
        n_total = contingency.loc[label].sum()
        n_yes = contingency.loc[label, 'Yes']
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'n={n_yes}/{n_total}\n({height:.1f}%)',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Data points after filtering: 137

--- Contingency Table (Autonomy vs Appeal) ---
has_appeal      Yes  No
autonomy_level         
High              2   3
Low              74  58

--- Statistical Results ---
Fisher's Exact Test p-value: 0.6556
Odds Ratio: 0.5225

--- Appeal Process Rates ---
autonomy_level
High    40.000000
Low     56.060606
Name: Yes, dtype: float64


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot (Vertical Bar Chart).
*   **Purpose:** The plot compares the prevalence (percentage) of a formal appeal process across two distinct categories of system autonomy ("High" and "Low").

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Autonomy Level"
    *   **Categories:** Two discrete categories labeled "High" and "Low".
*   **Y-Axis:**
    *   **Title:** "Percentage with Formal Appeal Process (%)"
    *   **Range:** 0 to 100.
    *   **Scale:** Linear scale with major tick marks every 20 units (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Comparison:** The bar representing "Low" autonomy is taller than the bar representing "High" autonomy.
*   **Tallest Bar:** The "Low" autonomy category shows that 56.1% of the systems have a formal appeal process.
*   **Shortest Bar:** The "High" autonomy category shows that 40.0% of the systems have a formal appeal process.
*   **Visual Pattern:** There is an apparent inverse relationship in this dataset where systems with lower autonomy levels are more likely to have a formal appeal process than those with high autonomy.

### 4. Annotations and Legends
*   **Text Annotations (Above Bars):**
    *   **High Autonomy Bar:** "n=2/5 (40.0%)". This indicates that out of a total sample size of 5 systems classified as high autonomy, 2 possessed a formal appeal process.
    *   **Low Autonomy Bar:** "n=74/132 (56.1%)". This indicates that out of a total sample size of 132 systems classified as low autonomy, 74 possessed a formal appeal process.
*   **Color Coding:** The bars are distinct in color (Red/Pink for High, Blue for Low) to visually separate the categories, though no separate legend box is required as the x-axis labels serve this purpose.

### 5. Statistical Insights
*   **Prevalence of Appeals:** A formal appeal process is more common in low autonomy systems (56.1%) compared to high autonomy systems (40.0%).
*   **Sample Size Disparity:** There is a massive disparity in sample sizes between the two groups. The "High" autonomy group has a very small sample size ($n=5$), whereas the "Low" autonomy group has a much more robust sample size ($n=132$).
*   **Reliability:** Due to the extremely small sample size for the "High" category ($n=5$), the 40% figure is statistically fragile. A change in the status of just one system in that group would swing the percentage by 20%. Therefore, while the chart suggests High Autonomy systems are less likely to have appeal processes, this conclusion should be treated with caution until more data on high autonomy systems is collected.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
