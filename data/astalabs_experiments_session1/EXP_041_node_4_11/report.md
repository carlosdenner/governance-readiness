# Experiment 41: node_4_11

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_11` |
| **ID in Run** | 41 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:50:00.976698+00:00 |
| **Runtime** | 180.2s |
| **Parent** | `node_3_5` |
| **Children** | None |
| **Creation Index** | 42 |

---

## Hypothesis

> There is a statistically significant positive correlation between the incident
date and the attack complexity (technique count), reflecting the increasing
sophistication of adversarial AI threats over time.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6210 (Maybe True) |
| **Posterior** | 0.9029 (Definitely True) |
| **Surprise** | +0.3272 |
| **Surprise Interpretation** | Strong Positive (hypothesis strengthened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 22.0 |
| Uncertain | 1.0 |
| Maybe False | 7.0 |
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

**Objective:** Analyze the temporal trend of attack complexity in AI incidents.

### Steps
- 1. Load 'step3_enrichments.json'.
- 2. Parse 'incident_date' into datetime objects and convert to an ordinal format or numeric timestamp.
- 3. Extract 'technique_count' for each incident.
- 4. Perform a Pearson (linear) and Spearman (rank) correlation analysis between date and technique count.

### Deliverables
- Correlation coefficients (r/rho), p-values, and a scatter plot analysis.

---

## Analysis

The experiment successfully analyzed the temporal trend of AI incident
complexity using the 'step3_enrichments.json' dataset (n=52). The analysis
confirmed a statistically significant positive correlation between the incident
date and the number of attack techniques used.

Specific findings:
1. **Correlation:**
   - **Pearson correlation (r = 0.38, p = 0.006)** indicates a moderate linear
relationship.
   - **Spearman correlation (rho = 0.44, p = 0.001)** indicates a slightly
stronger monotonic relationship.
   - Both p-values are well below the alpha threshold of 0.05, confirming
statistical significance.

2. **Trend Analysis:** The linear regression slope (approx. 0.46 techniques per
year) suggests that the complexity of reported AI incidents is increasing over
time. The scatter plot visualizes this trajectory, starting from a baseline of
~4 techniques in 2016 and rising to ~9 techniques by 2026.

3. **Data Distribution:** The plot reveals a high concentration of incidents
between 2023 and 2026, consistent with the explosion of generative AI deployment
and scrutiny. The variance (heteroscedasticity) also increases, showing that
while average complexity is rising, the range of incident types (simple to
complex) is widening.

---

## Review

The experiment successfully tested the hypothesis regarding the temporal
evolution of AI attack complexity. By analyzing 52 MITRE ATLAS case studies from
'step3_enrichments.json', the study found a statistically significant positive
correlation between the incident date and the number of adversarial techniques
employed (Attack Complexity).

**Specific Findings:**
- **Pearson Correlation:** r = 0.3768 (p = 0.0059), indicating a moderate linear
increase.
- **Spearman Correlation:** rho = 0.4350 (p = 0.0013), confirming a monotonic
upward trend.
- **Trend:** The regression analysis shows an average increase of approximately
0.46 techniques per year, rising from a baseline of ~4 techniques in 2016 to ~9
techniques by 2026.

**Conclusion:** The hypothesis is **supported**. The data reflects a measurable
increase in the sophistication of reported adversarial AI threats over time,
with a notable acceleration in incident density and variance from 2023 onwards.

---

## Code

```python
import json
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import sys
import os

# Try to locate the file
filename = 'step3_enrichments.json'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    # Last ditch effort: search recursively or just fail
    print(f"Error: {filename} not found in current ({os.getcwd()}) or parent directory.")
    sys.exit(1)

try:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Ensure required columns exist
    if 'incident_date' not in df.columns or 'technique_count' not in df.columns:
        print("Error: Required columns 'incident_date' or 'technique_count' missing.")
        sys.exit(1)

    # Convert incident_date to datetime
    # Coerce errors to NaT to handle potential malformed dates, then drop them
    df['incident_date_dt'] = pd.to_datetime(df['incident_date'], errors='coerce')
    
    # Filter out rows with invalid dates or technique counts
    df_clean = df.dropna(subset=['incident_date_dt', 'technique_count']).copy()
    
    if len(df_clean) < 2:
        print("Insufficient data points for correlation analysis.")
        sys.exit(0)

    # Convert date to ordinal number for correlation calculation
    df_clean['date_numeric'] = df_clean['incident_date_dt'].apply(lambda x: x.toordinal())

    # Perform Correlation Analysis
    pearson_r, pearson_p = stats.pearsonr(df_clean['date_numeric'], df_clean['technique_count'])
    spearman_r, spearman_p = stats.spearmanr(df_clean['date_numeric'], df_clean['technique_count'])

    print("=== Temporal Analysis of Attack Complexity ===")
    print(f"File loaded: {filepath}")
    print(f"Number of incidents analyzed: {len(df_clean)}")
    print(f"Date Range: {df_clean['incident_date_dt'].min().date()} to {df_clean['incident_date_dt'].max().date()}")
    print("\n--- Correlation Results ---")
    print(f"Pearson Correlation (r): {pearson_r:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman Correlation (rho): {spearman_r:.4f} (p-value: {spearman_p:.4f})")

    # Interpretation
    alpha = 0.05
    significance = "statistically significant" if pearson_p < alpha else "not statistically significant"
    direction = "positive" if pearson_r > 0 else "negative"
        
    print(f"\nConclusion: There is a weak {direction} correlation which is {significance}.")

    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(df_clean['incident_date_dt'], df_clean['technique_count'], 
                color='#4c72b0', alpha=0.7, label='Incident', edgecolors='w', s=60)
    
    # Trend line
    z = np.polyfit(df_clean['date_numeric'], df_clean['technique_count'], 1)
    p = np.poly1d(z)
    
    # Plot trend line
    plt.plot(df_clean['incident_date_dt'], p(df_clean['date_numeric']), 
             color='#c44e52', linestyle='--', linewidth=2, 
             label=f'Trend (slope={z[0]:.2e})')
    
    plt.title('Temporal Trend of AI Incident Complexity (Technique Count)')
    plt.xlabel('Incident Date')
    plt.ylabel('Technique Count (Complexity)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Temporal Analysis of Attack Complexity ===
File loaded: step3_enrichments.json
Number of incidents analyzed: 52
Date Range: 2016-03-23 to 2026-02-03

--- Correlation Results ---
Pearson Correlation (r): 0.3768 (p-value: 0.0059)
Spearman Correlation (rho): 0.4350 (p-value: 0.0013)

Conclusion: There is a weak positive correlation which is statistically significant.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Scatter Plot with an overlaid Linear Regression Trend Line.
*   **Purpose:** The plot visualizes individual AI incidents over time to identify correlations between the date of the incident and its complexity (measured by the count of techniques involved). The trend line helps to visualize the general direction of this relationship over the sampled period.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Incident Date"
    *   **Range:** The axis spans from **2016 to roughly mid-2026**.
    *   **Units:** Time (Years).
*   **Y-Axis:**
    *   **Title:** "Technique Count (Complexity)"
    *   **Range:** The vertical axis displays values from **0 to 16** (with grid lines marked every 2 units).
    *   **Units:** Count (integer number of techniques).

### 3. Data Trends
*   **Overall Trend:** There is a clear **positive correlation** between time and incident complexity. As time progresses from 2016 to 2026, the complexity of AI incidents generally increases.
*   **Data Density/Clustering:**
    *   **Sparse Early Data:** There is very little data plotted between 2016 and 2019, with only a single data point visible in 2016.
    *   **Increasing Frequency:** There is a noticeable increase in the density of data points starting around 2020, with the densest cluster of incidents occurring between **2023 and 2026**. This suggests either an increase in the frequency of AI incidents or an improvement in the tracking/reporting of these events in later years.
*   **Variance:** The spread (variance) of the data increases over time. In the earlier years, points are few. By 2025-2026, the technique counts vary widely, ranging from as low as **1** to as high as **15-16**.
*   **Outliers:**
    *   **High Values:** The highest complexity recorded is **16** (occurring around late 2023) and **15** (occurring in 2026).
    *   **Low Values:** A notable low outlier occurs in mid-2025 with a technique count of just **1**.

### 4. Annotations and Legends
*   **Legend (Top Right):**
    *   **"Incident" (Blue Circle):** Identifies the individual data points representing specific AI incidents. The points are semi-transparent, allowing the viewer to see areas where data points overlap (indicating multiple incidents with the same date and complexity).
    *   **"Trend (slope=1.27e-03)" (Red Dashed Line):** Identifies the line of best fit.
*   **Slope Annotation:** The slope is explicitly noted as **1.27e-03** ($0.00127$).
    *   *Note on Slope:* Given the visual rise of the line (rising roughly 5 units over 10 years), this slope value is likely calculated based on a daily timestamp unit (approx. $0.00127 \times 365 \text{ days} \approx 0.46$ increase in technique count per year).

### 5. Statistical Insights
*   **Rising Complexity:** The linear regression model indicates a steady rise in the average complexity of AI incidents. The trend line starts at a technique count of roughly **4.2 in 2016** and rises to approximately **9 in 2026**.
*   **Heteroscedasticity:** The data exhibits heteroscedasticity, meaning the variability of the "Technique Count" is not constant; it increases as time goes on. While the average complexity is rising, the range of possible complexities is expanding, indicating that while we are seeing much more complex incidents, simple incidents are still occurring.
*   **Incident Acceleration:** The visual clustering suggests a sharp acceleration in the number of reported incidents roughly after 2023, highlighting a period of increased activity or scrutiny in the AI domain.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
