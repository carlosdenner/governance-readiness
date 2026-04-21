# Experiment 17: node_4_2

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_2` |
| **ID in Run** | 17 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:01:23.899051+00:00 |
| **Runtime** | 170.2s |
| **Parent** | `node_3_3` |
| **Children** | `node_5_24` |
| **Creation Index** | 18 |

---

## Hypothesis

> There is a statistically significant positive temporal trend in AI incident
complexity, with later incidents involving a higher number of distinct attack
techniques.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7177 (Likely True) |
| **Posterior** | 0.9277 (Definitely True) |
| **Surprise** | +0.2437 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 27.0 |
| Uncertain | 3.0 |
| Maybe False | 0.0 |
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

**Objective:** Test for an increase in attack complexity over time.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Convert 'incident_date' to datetime objects (handling various formats).
- 3. Extract a numerical representation of the date (e.g., ordinal or timestamp).
- 4. Perform a Pearson or Spearman correlation analysis between 'incident_date' and 'technique_count'.
- 5. Fit a linear regression model to visualize the trend.

### Deliverables
- Scatter plot with regression line and correlation coefficient (r/rho, p-value).

---

## Analysis

The experiment successfully validated the hypothesis that AI incident complexity
is increasing over time. Analysis of 52 incidents from
'step3_incident_coding.csv' spanning 2016 to 2026 revealed a statistically
significant positive correlation between the incident date and the number of
attack techniques used. The Pearson correlation coefficient was r=0.38
(p=0.0059), and the Spearman rank correlation was rho=0.44 (p=0.0013), both
indicating a moderate but significant upward trend. The linear regression model
confirms that the average complexity of attacks has roughly doubled over the
observed decade, rising from ~4 techniques in 2016 to ~9 in 2026. This suggests
that adversarial actors are evolving their capabilities, employing increasingly
sophisticated, multi-step attack chains (e.g., combining reconnaissance, initial
access, and impact techniques) rather than simple, single-technique exploits.

---

## Review

The experiment successfully validated the hypothesis that AI incident complexity
is increasing over time. Analysis of 52 incidents from
'step3_incident_coding.csv' spanning 2016 to 2026 revealed a statistically
significant positive correlation between the incident date and the number of
attack techniques used. The Pearson correlation coefficient was r=0.38
(p=0.0059) and the Spearman rank correlation was rho=0.44 (p=0.0013), both
indicating a moderate but significant upward trend. The linear regression model
confirms that the average complexity of attacks has roughly doubled over the
observed decade, rising from ~4 techniques in 2016 to ~9 in 2026. This suggests
that adversarial actors are evolving their capabilities, employing increasingly
sophisticated, multi-step attack chains rather than simple exploits.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os
import sys

# Define file path based on instructions (one level above)
file_path = '../step3_incident_coding.csv'

# Fallback to current directory if not found (handling environment variations)
if not os.path.exists(file_path):
    file_path = 'step3_incident_coding.csv'

# Load the dataset
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded dataset from: {file_path}")
except FileNotFoundError:
    print(f"Error: Could not find dataset at {file_path}")
    sys.exit(1)

# Preprocessing Dates
# 'incident_date' in ATLAS can be Year, Year-Month, or Full Date. 
# pd.to_datetime handles these, defaulting to Jan 1st for Year-only.
df['date_obj'] = pd.to_datetime(df['incident_date'], errors='coerce')

# Check for invalid dates
initial_count = len(df)
df = df.dropna(subset=['date_obj'])
dropped_count = initial_count - len(df)
if dropped_count > 0:
    print(f"Dropped {dropped_count} rows with invalid dates.")

# Ensure technique_count is numeric
df['technique_count'] = pd.to_numeric(df['technique_count'], errors='coerce')
df = df.dropna(subset=['technique_count'])

# Sort by date
df = df.sort_values('date_obj')

# --- Statistical Analysis ---

# Convert date to ordinal (integer) for correlation analysis
df['date_ordinal'] = df['date_obj'].apply(lambda x: x.toordinal())

x = df['date_ordinal']
y = df['technique_count']

if len(df) < 2:
    print("Insufficient data points for analysis.")
    sys.exit(0)

# Pearson Correlation (Linear)
pearson_r, pearson_p = stats.pearsonr(x, y)

# Spearman Correlation (Monotonic - better if trend is non-linear or data is ordinal-like)
spearman_r, spearman_p = stats.spearmanr(x, y)

# Linear Regression for the trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Interpret trend
trend_direction = "Positive" if slope > 0 else "Negative"
significance = "Significant" if p_value < 0.05 else "Not Significant"

print("\n=== Temporal Trend Analysis of Incident Complexity ===")
print(f"Incidents Analyzed: {len(df)}")
print(f"Date Range: {df['date_obj'].min().date()} to {df['date_obj'].max().date()}")
print(f"Correlation (Pearson): r={pearson_r:.4f}, p={pearson_p:.4f}")
print(f"Correlation (Spearman): rho={spearman_r:.4f}, p={spearman_p:.4f}")
print(f"Linear Trend Slope: {slope:.5f} techniques/day")
print(f"Conclusion: {trend_direction} trend ({significance})")

# --- Visualization ---
plt.figure(figsize=(10, 6))

# Scatter plot of incidents
plt.scatter(df['date_obj'], df['technique_count'], color='blue', alpha=0.6, label='Incidents')

# Plot Regression Line
# Create a sequence of dates for the line to look smooth or just use min/max
x_line_dates = df['date_obj']
y_line_pred = slope * df['date_ordinal'] + intercept

plt.plot(x_line_dates, y_line_pred, color='red', linewidth=2, label=f'Trend (r={pearson_r:.2f})')

plt.title('Temporal Trend of AI Incident Complexity (Technique Count)')
plt.xlabel('Incident Date')
plt.ylabel('Complexity (Number of Techniques Used)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Improve date formatting on x-axis
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded dataset from: step3_incident_coding.csv

=== Temporal Trend Analysis of Incident Complexity ===
Incidents Analyzed: 52
Date Range: 2016-03-23 to 2026-02-03
Correlation (Pearson): r=0.3768, p=0.0059
Correlation (Spearman): rho=0.4350, p=0.0013
Linear Trend Slope: 0.00127 techniques/day
Conclusion: Positive trend (Significant)


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Scatter Plot with a Linear Regression (Trend) Line.
*   **Purpose:** The plot visualizes the relationship between time (dates of incidents) and the complexity of those incidents (measured by the count of techniques used). It aims to determine if AI incidents are becoming more complex over time.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Incident Date"
    *   **Range:** The axis spans from the beginning of **2016** through mid-**2026**.
    *   **Format:** Years are displayed at regular intervals, titled to the right.
*   **Y-Axis:**
    *   **Label:** "Complexity (Number of Techniques Used)"
    *   **Range:** Ticks range from **2 to 16**, though the visible plot area accommodates values slightly below 2 and up to roughly 17.
    *   **Units:** Integer count of techniques.

### 3. Data Trends
*   **Overall Trend:** There is a visible upward trend. As time progresses from 2016 to 2026, the complexity of incidents generally increases.
*   **Distribution/Density:**
    *   **2016–2019:** Data is very sparse, with only a single visible data point around late 2016.
    *   **2020–2022:** The frequency of incidents increases, with data points clustered mostly between complexity levels of 4 and 10.
    *   **2023–2026:** There is a significant increase in the density of data points. The spread (variance) of complexity also widens significantly during this period.
*   **Tallest/Highest Value:** The highest complexity recorded is **16 techniques**, occurring in early 2024. Another high-complexity incident (15 techniques) appears in 2026.
*   **Lowest Value:** The lowest complexity is **1 technique**, appearing in late 2025.

### 4. Annotations and Legends
*   **Title:** "Temporal Trend of AI Incident Complexity (Technique Count)"
*   **Legend (Top Left):**
    *   **Blue Circle ("Incidents"):** Represents individual AI incident data points. The dots are semi-transparent, allowing the viewer to see where multiple points overlap.
    *   **Red Line ("Trend (r=0.38)"):** Indicates the linear regression line showing the general direction of the data.
*   **Grid:** A light, dashed grid is applied to the background to assist in reading specific x and y values.

### 5. Statistical Insights
*   **Correlation (r=0.38):** The correlation coefficient ($r$) of 0.38 indicates a **weak to moderate positive correlation**. While there is a tendency for complexity to increase over time, the relationship is not perfectly linear, and there is significant variation that time alone does not explain.
*   **Increasing Variance (Heteroscedasticity):** The vertical spread of the data points increases over time. In earlier years, points are closer to the trend line. By 2024-2026, the points range wildly from 1 to 16. This suggests that while average complexity is rising, the *variability* of incident complexity is also increasing—we are seeing both very simple and extremely complex incidents occurring simultaneously in later years.
*   **Projected Growth:** The red trend line starts at a complexity level of approximately 4.2 in 2016 and rises to approximately 9 by 2026, suggesting the "average" incident complexity has roughly doubled over the decade shown.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
