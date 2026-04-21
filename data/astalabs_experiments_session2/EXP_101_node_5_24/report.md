# Experiment 101: node_5_24

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_24` |
| **ID in Run** | 101 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:54:02.499077+00:00 |
| **Runtime** | 343.8s |
| **Parent** | `node_4_1` |
| **Children** | `node_6_41` |
| **Creation Index** | 102 |

---

## Hypothesis

> The Autonomy-Risk Escalation: Higher levels of AI autonomy are statistically
associated with 'Tangible' harms (e.g., physical, financial) rather than
'Intangible' harms, indicating that loss of human control correlates with safety
risks.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4176 (Maybe False) |
| **Surprise** | -0.3892 |
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
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Investigate the relationship between system autonomy levels and the tangibility of harm using specific dataset mappings.

### Steps
- 1. Load the `astalabs_discovery_all_data.csv` dataset and filter for rows where `source_table` is 'aiid_incidents'.
- 2. Create a new variable `Autonomy_Bin` by mapping the `Autonomy Level` column: assign 'Autonomy1' to 'Low Autonomy' and combine 'Autonomy2' and 'Autonomy3' into 'High Autonomy'.
- 3. Create a new variable `Harm_Bin` by mapping the `Tangible Harm` column: assign 'tangible harm definitively occurred' to 'Tangible Harm' and 'no tangible harm, near-miss, or issue' to 'Intangible Harm'. Exclude other values (e.g., near-miss risks) to focus on actualized incidents.
- 4. Drop rows where either `Autonomy_Bin` or `Harm_Bin` is null.
- 5. Generate a Contingency Table of `Autonomy_Bin` vs. `Harm_Bin`.
- 6. Perform a Chi-Square Test of Independence to assess statistical significance.
- 7. Calculate row percentages to compare the proportion of Tangible Harm across Autonomy levels.

### Deliverables
- 1. Contingency Table of Autonomy vs. Harm Type.
- 2. Chi-Square Test Statistics (Chi2, p-value).
- 3. Stacked Bar Chart visualizing the distribution of Tangible vs. Intangible harm by Autonomy level.

---

## Analysis

The experiment successfully analyzed 155 valid AIID incidents to test the
'Autonomy-Risk Escalation' hypothesis.

**Findings:**
1.  **Hypothesis Not Supported:** The Chi-Square test yielded a p-value of 0.23,
indicating no statistically significant association between AI autonomy levels
and the nature of harm (Tangible vs. Intangible). The null hypothesis cannot be
rejected.
2.  **Directional Trend:** Although not statistically significant, the data
showed a directional trend aligning with the hypothesis. High Autonomy systems
had a higher proportion of 'Tangible Harm' (48.4%) compared to Low Autonomy
systems (37.4%).
3.  **Prevalence of Intangible Harm:** 'Intangible Harm' (e.g., bias,
reputation, civil rights) remains the dominant failure mode across all systems,
accounting for 62.6% of Low Autonomy incidents and 51.6% of High Autonomy
incidents.

**Conclusion:** While increasing autonomy appears to shift the risk profile
slightly toward tangible consequences, the difference is not statistically
robust in this sample. Governance frameworks must prioritize intangible risks
regardless of the system's autonomy level.

---

## Review

The experiment successfully analyzed 155 valid AIID incidents to test the
'Autonomy-Risk Escalation' hypothesis using the corrected data mappings.

**Findings:**
1.  **Hypothesis Not Supported:** The Chi-Square test yielded a p-value of 0.23,
indicating no statistically significant association between AI autonomy levels
and the nature of harm (Tangible vs. Intangible). The null hypothesis cannot be
rejected.
2.  **Directional Trend:** Although not statistically significant, the data
showed a directional trend aligning with the hypothesis. High Autonomy systems
had a higher proportion of 'Tangible Harm' (48.4%) compared to Low Autonomy
systems (37.4%).
3.  **Prevalence of Intangible Harm:** 'Intangible Harm' (e.g., bias,
reputation, civil rights) remains the dominant failure mode across all systems,
accounting for 62.6% of Low Autonomy incidents and 51.6% of High Autonomy
incidents.

**Conclusion:** While increasing autonomy appears to shift the risk profile
slightly toward tangible consequences, the difference is not statistically
robust in this sample. Governance frameworks must prioritize intangible risks
regardless of the system's autonomy level.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

def run_experiment():
    print("Starting Experiment: Autonomy-Risk Escalation Analysis (Corrected Mappings)...")

    # 1. Load Dataset
    file_path = 'astalabs_discovery_all_data.csv'
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Failed to load csv: {e}")
        return

    # 2. Filter for AIID Incidents
    df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents found: {len(df_aiid)}")

    # 3. Define Mappings based on specific dataset values
    
    # Autonomy Mapping
    # Autonomy1 -> Low
    # Autonomy2, Autonomy3 -> High
    def map_autonomy_corrected(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if s == 'Autonomy1':
            return 'Low Autonomy'
        elif s in ['Autonomy2', 'Autonomy3']:
            return 'High Autonomy'
        return None

    # Harm Mapping
    # 'tangible harm definitively occurred' -> Tangible
    # 'no tangible harm, near-miss, or issue' -> Intangible
    # Others -> None (Excluded)
    def map_harm_corrected(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if s == 'tangible harm definitively occurred':
            return 'Tangible Harm'
        elif s == 'no tangible harm, near-miss, or issue':
            return 'Intangible Harm'
        # Excluding near-misses and risks as per experiment plan
        return None

    # Apply mappings
    # Note: Column names identified in previous step: 'Autonomy Level' and 'Tangible Harm'
    df_aiid['Autonomy_Bin'] = df_aiid['Autonomy Level'].apply(map_autonomy_corrected)
    df_aiid['Harm_Bin'] = df_aiid['Tangible Harm'].apply(map_harm_corrected)

    # 4. Filter Data
    df_clean = df_aiid.dropna(subset=['Autonomy_Bin', 'Harm_Bin'])
    print(f"Records available for analysis after cleaning: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("No records matched the criteria. Dumping sample values for debugging:")
        print("Autonomy:", df_aiid['Autonomy Level'].unique()[:5])
        print("Harm:", df_aiid['Tangible Harm'].unique()[:5])
        return

    # 5. Statistical Analysis (Chi-Square)
    contingency_table = pd.crosstab(df_clean['Autonomy_Bin'], df_clean['Harm_Bin'])
    print("\n--- Contingency Table ---")
    print(contingency_table)

    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Statistically Significant association found (Reject H0)")
    else:
        print("Result: No statistically significant association found (Fail to reject H0)")

    # 6. Visualization
    # Calculate percentages for the stacked bar chart
    row_props = pd.crosstab(df_clean['Autonomy_Bin'], df_clean['Harm_Bin'], normalize='index') * 100
    
    print("\n--- Row Percentages ---")
    print(row_props)

    plt.figure(figsize=(10, 6))
    ax = row_props.plot(kind='bar', stacked=True, color=['#99ccff', '#ff9999'], edgecolor='black')
    
    plt.title('Distribution of Harm Types by AI Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Percentage of Incidents')
    plt.xticks(rotation=0)
    plt.legend(title='Harm Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Annotate bars
    for c in ax.containers:
        # Filter out labels for very small segments to avoid clutter
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Experiment: Autonomy-Risk Escalation Analysis (Corrected Mappings)...
AIID Incidents found: 1362
Records available for analysis after cleaning: 155

--- Contingency Table ---
Harm_Bin       Intangible Harm  Tangible Harm
Autonomy_Bin                                 
High Autonomy               33             31
Low Autonomy                57             34

--- Chi-Square Test Results ---
Chi2 Statistic: 1.4652
P-value: 0.2261
Result: No statistically significant association found (Fail to reject H0)

--- Row Percentages ---
Harm_Bin       Intangible Harm  Tangible Harm
Autonomy_Bin                                 
High Autonomy        51.562500      48.437500
Low Autonomy         62.637363      37.362637


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** The plot is designed to compare the relative distribution (percentage) of two distinct categories of harm ("Intangible" vs. "Tangible") across two different groups defined by AI autonomy levels ("High Autonomy" and "Low Autonomy"). By normalizing the bars to 100%, it focuses on the proportion of each harm type within each autonomy setting rather than the absolute number of incidents.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Autonomy Level"
    *   **Categories:** The axis displays two categorical groups: "High Autonomy" and "Low Autonomy."
*   **Y-Axis:**
    *   **Label:** "Percentage of Incidents"
    *   **Range:** The scale runs from 0 to 100.
    *   **Units:** Percentage (%), indicated by the values reaching 100 and the percent signs within the data labels.

### 3. Data Trends
*   **Dominant Category:** In both autonomy levels, "Intangible Harm" (blue) constitutes the majority of incidents (>50%).
*   **High Autonomy:**
    *   The distribution is fairly balanced but leans slightly toward "Intangible Harm" at **51.6%**, while "Tangible Harm" accounts for **48.4%**.
*   **Low Autonomy:**
    *   There is a more pronounced skew. "Intangible Harm" rises significantly to **62.6%**, whereas "Tangible Harm" drops to **37.4%**.
*   **Comparison Pattern:** As the level of autonomy decreases (moving from High to Low), the proportion of intangible harms increases by 11 percentage points, while the proportion of tangible harms decreases correspondingly.

### 4. Annotations and Legends
*   **Legend:** A box in the upper right corner titled "Harm Category" defines the color coding:
    *   **Light Blue:** Represents "Intangible Harm."
    *   **Light Red/Pink:** Represents "Tangible Harm."
*   **Data Labels:** Numerical percentage values are annotated directly inside each bar segment to provide precise data points (e.g., "48.4%" and "51.6%" for High Autonomy; "37.4%" and "62.6%" for Low Autonomy).
*   **Title:** The chart is titled "Distribution of Harm Types by AI Autonomy Level," clearly stating the subject of the analysis.

### 5. Statistical Insights
*   **Shift in Risk Profile:** The data suggests a correlation between the level of AI autonomy and the nature of the harm produced. Higher autonomy in AI systems is associated with a greater relative risk of **Tangible Harm** (physical or concrete damage) compared to low autonomy systems.
*   **Prevalence of Intangible Harm:** Regardless of autonomy level, **Intangible Harm** (likely referring to reputational, psychological, or societal damage) remains the primary mode of failure, accounting for the majority of incidents in both scenarios.
*   **Implication:** This visualization implies that while increasing AI autonomy introduces more tangible risks, the reduction of autonomy does not eliminate harm; rather, it shifts the distribution heavily toward intangible issues.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
