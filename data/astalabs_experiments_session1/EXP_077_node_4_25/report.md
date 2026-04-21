# Experiment 77: node_4_25

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_25` |
| **ID in Run** | 77 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:01:51.139333+00:00 |
| **Runtime** | 235.0s |
| **Parent** | `node_3_12` |
| **Children** | None |
| **Creation Index** | 78 |

---

## Hypothesis

> The number of backing literature citations for a competency statement is
positively correlated with the number of applicable architecture controls,
suggesting that more complex technical requirements (higher control fan-out) are
supported by a broader evidentiary base.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7339 (Likely True) |
| **Posterior** | 0.5661 (Maybe True) |
| **Surprise** | -0.1947 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
| Uncertain | 1.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 42.0 |
| Uncertain | 9.0 |
| Maybe False | 39.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess the correlation between evidence volume and technical complexity.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Create variable 'citation_count': Count the number of citations (regex pattern '\[#\d+\]') in the 'competency_statement' text or 'evidence_summary'.
- 3. Create variable 'control_count': Count the number of semicolon-separated items in the 'applicable_controls' column.
- 4. Perform a Pearson (or Spearman) correlation test between 'citation_count' and 'control_count'.
- 5. Visualize with a scatter plot and a regression line.

### Deliverables
- Correlation coefficient, p-value, and scatter plot.

---

## Analysis

The experiment successfully loaded the 'step2_competency_statements.csv' dataset
and calculated the 'citation_count' (evidence volume) and 'control_count'
(technical complexity) for all 42 competency statements.

The correlation analysis revealed a weak positive relationship between the
number of literature citations and the number of mapped architecture controls.
Specifically, the Pearson correlation coefficient was 0.2675 (p=0.0868) and the
Spearman rank correlation was 0.2624 (p=0.0932).

Since both p-values exceed the standard significance threshold of 0.05, we fail
to reject the null hypothesis. While there is a directional trend suggesting
that heavily cited competencies might map to slightly more controls, the
evidence is not statistically significant. The generated scatter plot visualizes
this weak relationship with a regression line surrounded by a wide confidence
interval, further indicating the lack of a robust predictive link between
evidence volume and control fan-out in this dataset.

---

## Review

The experiment was successfully executed and faithfully followed the proposed
plan. The code correctly loaded the "step2_competency_statements.csv" dataset
(after an initial path correction), engineered the required features
("citation_count" and "control_count"), and performed the specified correlation
analysis.

The results indicate a weak positive correlation between the volume of backing
literature and the number of mapped architecture controls (Pearson r=0.2675,
Spearman rho=0.2624). However, with p-values of 0.0868 and 0.0932 respectively,
this relationship is not statistically significant at the standard alpha=0.05
level. Consequently, the analysis provides sufficient evidence to fail to reject
the null hypothesis, suggesting that within this framework, the depth of
evidentiary support does not reliably predict the breadth of technical
implementation requirements. The generated scatter plot accurately visualizes
this weak, non-significant trend.

---

## Code

```python
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

def load_dataset(filename):
    # Try current directory first
    if os.path.exists(filename):
        return pd.read_csv(filename)
    # Try one level up
    parent_path = os.path.join('..', filename)
    if os.path.exists(parent_path):
        return pd.read_csv(parent_path)
    raise FileNotFoundError(f"Could not find {filename} in . or ..")

try:
    # 1. Load the dataset
    df = load_dataset('step2_competency_statements.csv')
    print(f"Dataset loaded successfully with shape: {df.shape}")

    # 2. Create variable 'citation_count'
    # Pattern to match citations like [#1], [#12]
    citation_pattern = r'\[#\d+\]'
    
    def count_citations(row):
        # Concatenate text from both columns to search for citations
        text = str(row['competency_statement']) + " " + str(row['evidence_summary'])
        # Find all occurrences
        matches = re.findall(citation_pattern, text)
        return len(matches)

    df['citation_count'] = df.apply(count_citations, axis=1)

    # 3. Create variable 'control_count'
    def count_controls(val):
        if pd.isna(val) or str(val).strip() == '':
            return 0
        # Split by semicolon, strip whitespace, filter out empty strings
        items = [x.strip() for x in str(val).split(';') if x.strip()]
        return len(items)

    df['control_count'] = df['applicable_controls'].apply(count_controls)

    # Print sample to verify
    print("\nSample of calculated counts:")
    print(df[['competency_id', 'citation_count', 'control_count']].head())

    # 4. Perform Correlation Tests
    # Pearson (Linear)
    pearson_r, pearson_p = stats.pearsonr(df['citation_count'], df['control_count'])
    # Spearman (Rank - robust to outliers/non-normal)
    spearman_rho, spearman_p = stats.spearmanr(df['citation_count'], df['control_count'])

    print("\n=== Correlation Results ===")
    print(f"Pearson Correlation (r): {pearson_r:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman Correlation (rho): {spearman_rho:.4f} (p-value: {spearman_p:.4f})")

    # Interpretation
    alpha = 0.05
    if pearson_p < alpha:
        print("Conclusion: Statistically significant linear correlation found.")
    else:
        print("Conclusion: No statistically significant linear correlation found.")

    # 5. Visualize
    plt.figure(figsize=(10, 6))
    sns.regplot(x='citation_count', y='control_count', data=df, 
                scatter_kws={'alpha':0.6, 's':60}, line_kws={'color':'red'})
    
    plt.title('Correlation: Literature Evidence vs. Technical Complexity')
    plt.xlabel('Citation Count (Evidence Volume)')
    plt.ylabel('Control Count (Architecture Fan-out)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add correlation text to plot
    plt.annotate(f'Pearson r={pearson_r:.2f} (p={pearson_p:.3f})',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Dataset loaded successfully with shape: (42, 8)

Sample of calculated counts:
  competency_id  citation_count  control_count
0       COMP-01               5              3
1       COMP-02               4              1
2       COMP-03               4              1
3       COMP-04               4              1
4       COMP-05               4              1

=== Correlation Results ===
Pearson Correlation (r): 0.2675 (p-value: 0.0868)
Spearman Correlation (rho): 0.2624 (p-value: 0.0932)
Conclusion: No statistically significant linear correlation found.


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Scatter plot with a linear regression line and a confidence interval band.
*   **Purpose:** The plot visualizes the relationship (correlation) between two variables: "Citation Count" and "Control Count." It attempts to determine if an increase in evidence volume correlates with an increase in architectural fan-out.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Citation Count (Evidence Volume)"
    *   **Range:** The axis displays values from roughly 1.8 to 6.2, with major tick marks at intervals of 0.5 (2.0 to 6.0).
*   **Y-Axis:**
    *   **Label:** "Control Count (Architecture Fan-out)"
    *   **Range:** The axis displays values from approximately 0.9 to 3.1, with tick marks appearing at intervals of 0.25 (1.00 to 3.00).

### 3. Data Trends
*   **Distribution:** The data points (blue circles) appear to be discrete values rather than continuous, particularly on the Y-axis where points align perfectly with the integers 1.0, 2.0, and 3.0. The X-axis data also clusters around integer values (2, 3, 4, 5, 6).
*   **Clustering/Overlap:** There appears to be overlapping data points (overplotting), indicated by the varying intensity of the blue color. Darker blue dots suggest multiple data points occupy the same coordinate (e.g., at X=2, Y=2 and X=4, Y=1).
*   **Direction:** The red regression line shows a positive slope, indicating a general trend where higher citation counts are associated with higher control counts.
*   **Spread:** The data is quite spread out vertically. for almost every X-value (citation count), there are data points spanning the full range of Y-values (1 to 3). This indicates high variance.

### 4. Annotations and Legends
*   **Title:** "Correlation: Literature Evidence vs. Technical Complexity"
*   **Statistical Annotation:** A text box in the top-left corner displays the statistical results: `Pearson r=0.27 (p=0.087)`.
*   **Regression Elements:**
    *   **Red Line:** Represents the linear line of best fit.
    *   **Pink Shaded Area:** Represents the confidence interval (likely 95%) around the regression line. The band widens slightly at the ends (X=6), indicating strictly less certainty about the trend at the extremes of the data range.

### 5. Statistical Insights
*   **Correlation Strength (r=0.27):** The Pearson correlation coefficient of 0.27 suggests a **weak positive correlation**. While there is a tendency for "Control Count" to increase as "Citation Count" increases, the relationship is not strong.
*   **Significance (p=0.087):** The p-value is 0.087. In most scientific contexts (where the threshold is typically p < 0.05), this result is **not statistically significant**. This means that there is an 8.7% probability that this pattern occurred by random chance, and we cannot confidently reject the null hypothesis that there is no relationship between these variables.
*   **Conclusion:** While the visual trend suggests that higher evidence volume (citations) might be linked to higher technical complexity (fan-out), the statistical evidence provided is inconclusive. More data might be needed to confirm if a true relationship exists.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
