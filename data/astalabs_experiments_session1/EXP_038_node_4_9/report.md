# Experiment 38: node_4_9

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_9` |
| **ID in Run** | 38 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:41:27.704287+00:00 |
| **Runtime** | 169.7s |
| **Parent** | `node_3_7` |
| **Children** | None |
| **Creation Index** | 39 |

---

## Hypothesis

> The complexity of an AI attack, measured by the number of adversarial techniques
employed, is positively correlated with the breadth of the organizational
competency gap (number of missing sub-competencies).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6048 (Maybe True) |
| **Posterior** | 0.8988 (Likely True) |
| **Surprise** | +0.3411 |
| **Surprise Interpretation** | Strong Positive (hypothesis strengthened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 21.0 |
| Uncertain | 1.0 |
| Maybe False | 8.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 90.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if multi-step attacks expose a wider range of organizational deficits than simpler attacks.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Extract 'technique_count' (numerical).
- 3. Parse 'sub_competency_ids' to calculate the 'Gap Breadth' (count of distinct IDs listed per row).
- 4. Perform a Pearson (or Spearman) correlation test between Technique Count and Gap Breadth.
- 5. Generate a scatter plot with a regression line.

### Deliverables
- Correlation coefficient, p-value, and scatter plot of Technique Count vs. Gap Breadth.

---

## Analysis

The experiment successfully validated the hypothesis that attack complexity
(measured by technique count) is strongly correlated with the breadth of
organizational competency gaps. Analysis of the 52 MITRE ATLAS incidents in
`step3_incident_coding.csv` revealed a strong, statistically significant
positive correlation (Pearson r=0.83, p<0.001; Spearman rho=0.76, p<0.001). The
data shows that as attacks become more complex (Mean techniques = 7.54), they
expose a significantly higher number of missing sub-competencies (Mean gaps =
4.81), rather than failing on a single control. This supports the theoretical
proposition that complex AI adversarial chains leverage systemic organizational
fragility, requiring a defense-in-depth approach that is currently lacking in
the observed cases.

---

## Review

The experiment was executed perfectly. The code correctly loaded the incident
dataset, parsed the necessary columns to calculate 'Gap Breadth', and performed
the requested statistical correlation tests. The results (Pearson r=0.83,
p<0.001) provide very strong evidence supporting the hypothesis.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Define the file path
filename = 'step3_incident_coding.csv'

# Check if file exists in current directory or up one level
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    # Fallback to the list of files provided in the prompt context to find where they might be
    # Assuming they are in the current working directory based on previous turns usually
    filepath = filename

print(f"Loading dataset from: {filepath}")

try:
    df = pd.read_csv(filepath)
    
    # Verify required columns exist
    required_columns = ['technique_count', 'sub_competency_ids']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols}")
        # Attempt to calculate technique_count if missing but techniques_used exists
        if 'technique_count' in missing_cols and 'techniques_used' in df.columns:
            print("Calculated technique_count from techniques_used.")
            df['technique_count'] = df['techniques_used'].fillna('').apply(lambda x: len(x.split(';')) if x else 0)
        else:
            raise ValueError(f"Cannot proceed without columns: {missing_cols}")

    # Calculate Gap Breadth (count of sub_competency_ids)
    # Handle NaNs and empty strings
    def count_gaps(val):
        if pd.isna(val) or val == '':
            return 0
        # Split by semicolon, strip whitespace, remove empty strings
        items = [x.strip() for x in str(val).split(';') if x.strip()]
        return len(items)

    df['gap_breadth'] = df['sub_competency_ids'].apply(count_gaps)

    # Extract vectors
    x = df['technique_count']
    y = df['gap_breadth']

    # Statistical Analysis
    pearson_corr, pearson_p = stats.pearsonr(x, y)
    spearman_corr, spearman_p = stats.spearmanr(x, y)

    print("\n--- Statistical Analysis ---")
    print(f"N = {len(df)}")
    print(f"Technique Count: Mean={x.mean():.2f}, Std={x.std():.2f}, Min={x.min()}, Max={x.max()}")
    print(f"Gap Breadth:     Mean={y.mean():.2f}, Std={y.std():.2f}, Min={y.min()}, Max={y.max()}")
    print("\nCorrelation Results:")
    print(f"Pearson correlation:  r={pearson_corr:.4f}, p={pearson_p:.4f}")
    print(f"Spearman correlation: rho={spearman_corr:.4f}, p={spearman_p:.4f}")

    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with jitter to handle overlapping points (since data is discrete counts)
    x_jitter = x + np.random.normal(0, 0.1, size=len(x))
    y_jitter = y + np.random.normal(0, 0.1, size=len(y))
    
    plt.scatter(x_jitter, y_jitter, alpha=0.6, label='Incident Data (Jittered)')

    # Regression Line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='red', label=f'Regression (r={r_value:.2f})')

    plt.title('Attack Complexity vs. Competency Gap Breadth')
    plt.xlabel('Technique Count (Complexity)')
    plt.ylabel('Gap Breadth (Missing Sub-competencies)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_incident_coding.csv

--- Statistical Analysis ---
N = 52
Technique Count: Mean=7.54, Std=2.93, Min=1, Max=16
Gap Breadth:     Mean=4.81, Std=1.40, Min=1, Max=9

Correlation Results:
Pearson correlation:  r=0.8320, p=0.0000
Spearman correlation: rho=0.7613, p=0.0000


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Scatter plot with an overlaid linear regression line.
*   **Purpose:** The plot visualizes the relationship and correlation between two variables: "Attack Complexity" (independent variable) and "Competency Gap Breadth" (dependent variable).

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Title:** "Technique Count (Complexity)"
    *   **Unit/Context:** Represents a count of techniques used, serving as a proxy for complexity.
    *   **Range:** The axis ticks range from 2 to 16. The data points span from approximately x=1 to x=16.
*   **Y-Axis (Vertical):**
    *   **Title:** "Gap Breadth (Missing Sub-competencies)"
    *   **Unit/Context:** Represents a count or score of missing sub-competencies.
    *   **Range:** The axis ticks range from 1 to 9. The data points span from y=1 to roughly y=9.

### 3. Data Trends
*   **Direction:** There is a clear **positive trend**. As the technique count (complexity) increases, the gap breadth (missing sub-competencies) also increases.
*   **Clustering:** The majority of the data points are clustered in the middle range, roughly between a technique count of 4 and 10, with gap breadths ranging between 3 and 6.
*   **Distribution:** The data points are somewhat spread out vertically at specific x-values (e.g., at x=6 and x=9), indicating variance in the gap breadth for attacks of similar complexity.
*   **Extremes:**
    *   The lowest data point is at approximately (1, 1).
    *   The highest data point is at approximately (16, 9).

### 4. Annotations and Legends
*   **Main Title:** "Attack Complexity vs. Competency Gap Breadth" positioned at the top center.
*   **Legend (Top Left):**
    *   **Blue Circle:** Labeled "Incident Data (Jittered)". "Jittered" implies that a small amount of random noise was added to the data points to prevent them from overlapping perfectly, allowing the viewer to see the density of data at discrete integer coordinates.
    *   **Red Line:** Labeled "Regression (r=0.83)". This indicates the red line is a linear fit model for the data.
*   **Grid:** A light gray, dashed grid is provided to help estimate values for individual data points.

### 5. Statistical Insights
*   **Strong Positive Correlation:** The correlation coefficient ($r$) is stated as **0.83**. In statistics, an $r$ value close to 1.0 indicates a strong positive linear relationship. This suggests that Attack Complexity is a strong predictor of Competency Gap Breadth.
*   **Interpretation:** The data indicates that as an attack becomes more complex (utilizing a higher count of techniques), the "Competency Gap" (the number of missing sub-competencies required to defend or respond) widens significantly.
*   **Linear Fit:** The red regression line passes centrally through the cloud of data points, suggesting that a linear model is a reasonably good fit for describing the relationship between these two variables.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
