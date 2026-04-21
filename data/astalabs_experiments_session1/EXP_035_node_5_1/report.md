# Experiment 35: node_5_1

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_1` |
| **ID in Run** | 35 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:37:35.714252+00:00 |
| **Runtime** | 179.4s |
| **Parent** | `node_4_6` |
| **Children** | `node_6_1` |
| **Creation Index** | 36 |

---

## Hypothesis

> Sub-competencies classified under 'Integration Readiness' have significantly
higher empirical validation (incident counts) than 'Trust Readiness' sub-
competencies, suggesting that technical failures are more readily observed or
reported than governance gaps.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7097 (Likely True) |
| **Posterior** | 0.1818 (Likely False) |
| **Surprise** | -0.6126 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 26.0 |
| Uncertain | 4.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Compare the distribution of incident counts between the two competency bundles.

### Steps
- 1. Load 'step3_coverage_map.csv'.
- 2. Group the data by 'bundle' ('Trust Readiness' vs. 'Integration Readiness').
- 3. Extract the 'incident_count' lists for both groups.
- 4. Perform a Mann-Whitney U test (non-parametric) to determine if there is a statistically significant difference in the median incident counts.

### Deliverables
- Boxplot of incident counts by bundle, descriptive statistics (median, mean, std), and Mann-Whitney U test results (U-statistic, p-value).

---

## Analysis

The experiment successfully compared the empirical validation (incident counts)
of 'Integration Readiness' versus 'Trust Readiness' sub-competencies using the
`step3_coverage_map.csv` dataset. The analysis revealed that the two bundles
have nearly identical incident distributions: Trust Readiness (Mean=5.63,
Median=3.5) and Integration Readiness (Mean=5.50, Median=4.5). The Mann-Whitney
U test yielded a U-statistic of 30.0 and a p-value of 0.6046, indicating no
statistically significant difference between the groups. Consequently, the
hypothesis that technical failures are more readily observed than governance
gaps is rejected. The boxplot visualization confirms that both bundles share
similar characteristics, including high variance and the presence of sub-
competencies with zero incident coverage.

---

## Review

The experiment successfully compared the empirical validation (incident counts)
of 'Integration Readiness' versus 'Trust Readiness' sub-competencies using the
`step3_coverage_map.csv` dataset. The analysis revealed that the two bundles
have nearly identical incident distributions: Trust Readiness (Mean=5.63,
Median=3.5) and Integration Readiness (Mean=5.50, Median=4.5). The Mann-Whitney
U test yielded a U-statistic of 30.0 and a p-value of 0.6046, indicating no
statistically significant difference between the groups. Consequently, the
hypothesis that technical failures are more readily observed than governance
gaps is rejected. The boxplot visualization confirms that both bundles share
similar characteristics, including high variance and the presence of sub-
competencies with zero incident coverage.

---

## Code

```python
import subprocess
import sys
import os

# Function to install packages if not found
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Import libraries with fallback installation
try:
    import pandas as pd
except ImportError:
    install('pandas')
    import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    install('matplotlib')
    import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    install('scipy')
    from scipy.stats import mannwhitneyu

# Load dataset
file_name = 'step3_coverage_map.csv'
file_path = f'../{file_name}'

# Check file existence and load
if not os.path.exists(file_path):
    if os.path.exists(file_name):
        file_path = file_name
    else:
        print(f"Error: {file_name} not found in parent or current directory.")
        sys.exit(1)

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Filter data by bundle
trust_data = df[df['bundle'] == 'Trust Readiness']
integration_data = df[df['bundle'] == 'Integration Readiness']

trust_counts = trust_data['incident_count']
integration_counts = integration_data['incident_count']

# Descriptive Statistics
print("\n--- Descriptive Statistics ---")
print(f"Trust Readiness (n={len(trust_counts)}):")
print(trust_counts.describe())
print(f"\nIntegration Readiness (n={len(integration_counts)}):")
print(integration_counts.describe())

# Mann-Whitney U Test
# Hypothesis: Integration > Trust (Alternative = 'greater')
stat, p_val = mannwhitneyu(integration_counts, trust_counts, alternative='greater')

print("\n--- Mann-Whitney U Test Results ---")
print(f"Hypothesis: Integration Readiness incident counts > Trust Readiness incident counts")
print(f"U-statistic: {stat}")
print(f"P-value: {p_val:.5f}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically significant difference detected.")
else:
    print("Result: No statistically significant difference detected.")

# Boxplot Visualization
plt.figure(figsize=(10, 6))
data_to_plot = [trust_counts, integration_counts]
labels = ['Trust Readiness', 'Integration Readiness']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
plt.title('Distribution of Incident Counts by Competency Bundle')
plt.ylabel('Number of Mapped Incidents')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add jittered points to show individual data points since N is small
import numpy as np
for i, data in enumerate(data_to_plot):
    y = data
    x = np.random.normal(1 + i, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.6)

plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_coverage_map.csv

--- Descriptive Statistics ---
Trust Readiness (n=8):
count     8.000000
mean      5.625000
std       6.545173
min       0.000000
25%       0.750000
50%       3.500000
75%       9.000000
max      19.000000
Name: incident_count, dtype: float64

Integration Readiness (n=8):
count     8.000000
mean      5.500000
std       6.141196
min       0.000000
25%       0.750000
50%       4.500000
75%       8.000000
max      18.000000
Name: incident_count, dtype: float64

--- Mann-Whitney U Test Results ---
Hypothesis: Integration Readiness incident counts > Trust Readiness incident counts
U-statistic: 30.0
P-value: 0.60461
Result: No statistically significant difference detected.

STDERR:
<ipython-input-1-b654a83c30e3>:80: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot with Overlaid Jitter/Strip Plot.
*   **Purpose:** This plot visualizes the distribution of numerical data ("Number of Mapped Incidents") across categorical groups ("Competency Bundle"). The box plot component summarizes the distribution statistics (median, quartiles, range), while the overlaid red dots represent individual data points, allowing the viewer to see the specific sample size and density of the data.

### 2. Axes
*   **X-axis:**
    *   **Label:** Represents the **Competency Bundle** categories.
    *   **Categories:** Two distinct groups: "Trust Readiness" and "Integration Readiness".
*   **Y-axis:**
    *   **Label:** "Number of Mapped Incidents".
    *   **Units:** Count (integer values).
    *   **Range:** The axis ticks range from 0.0 to 17.5, with the visual data extending slightly beyond, from a minimum of 0 to a maximum of approximately 19.

### 3. Data Trends
*   **Trust Readiness:**
    *   **Median:** The red line inside the box indicates a median incident count of approximately 3.5.
    *   **Spread:** The Interquartile Range (IQR), represented by the blue box, spans from roughly 1 to 9.
    *   **Range:** The whiskers extend from a minimum of 0 to a maximum of roughly 19.
    *   **Observations:** The data includes several points at the low end (0-1) and a single high value near 19, suggesting a wide variance.
*   **Integration Readiness:**
    *   **Median:** The median is slightly higher than the Trust Readiness group, sitting at approximately 4.5.
    *   **Spread:** The IQR is slightly more compressed than the Trust Readiness group, spanning from roughly 1 to 8.
    *   **Range:** The whiskers extend from 0 to approximately 18.
    *   **Observations:** Similar to the first group, there is a cluster of values near 0 and distinct high values near 8 and 18.

### 4. Annotations and Legends
*   **Red Dots:** represent individual data points (observations). Their horizontal position within each category is "jittered" (randomly shifted slightly left or right) to prevent points with the same value from overlapping, ensuring all data points are visible.
*   **Blue Box:** Represents the middle 50% of the data (Interquartile Range - Q1 to Q3).
*   **Red Line (inside box):** Indicates the Median value.
*   **Whiskers (Black lines):** Extend to the minimum and maximum values in the dataset.
*   **Grid Lines:** Horizontal dashed lines appear at intervals of 2.5 to aid in estimating values.

### 5. Statistical Insights
*   **Right-Skewed Distribution:** Both "Trust Readiness" and "Integration Readiness" show a right-skewed distribution (positive skew). The distance from the median to the maximum value is much larger than the distance from the median to the minimum value. This indicates that while most bundles have a low number of incidents (often close to zero), there are outlier cases with significantly higher incident counts (18-19).
*   **High Variability:** There is significant variability in both categories. Incident counts range from 0 all the way to nearly 20, suggesting that "readiness" in these contexts does not guarantee a uniform outcome regarding incident numbers.
*   **Similarity between Groups:** There is no statistically obvious difference between the two bundles based on visual inspection alone. Their medians, minimums, and maximums are very similar. "Integration Readiness" has a slightly higher median, but "Trust Readiness" has a slightly higher maximum and wider IQR.
*   **Zero-Inflation:** A notable number of data points (red dots) sit at or near zero for both categories, indicating that a significant portion of these bundles experienced no mapped incidents at all.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
